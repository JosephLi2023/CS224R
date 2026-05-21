# Turn-Level Reward Mechanism TODOs

This is the working checklist for experimenting with finalized turn-level reward mechanisms after the proposal alpha fix.

Current trainer semantics:

```text
A_H(i,t) = alpha * A_turn(i,t) + (1 - alpha) * A_traj(i)
```

So:

- `alpha = 0.0` is flat GRPO.
- `alpha = 1.0` is pure turn-level credit.
- All new decomposers should return a Python shape `[K][T_i]`, with no padded fake turns.

## Pipeline Contract

For any new method:

1. Add a decomposer in `src/algorithms/hgpo/decomposers/<method>.py`.
2. Add a branch in `src/algorithms/hgpo/decomposers/base.py::build_decomposer`.
3. Add config keys under `"hgpo": {"decomposer": "<method>"}` and a method-specific block if needed.
4. Return real turns only: `len(out[i]) == len(group.trajectories[i].turns)`.
5. Do not use zeros for missing/padded/unlabeled turns. Missing turns must be absent, not zero-filled.
6. Add a unit test that checks:
   - shape `[K][T_i]`;
   - no NaN;
   - variable-length trajectories;
   - nonzero positive and negative `A_turn` after `compute_turn_advantages`;
   - integration through `HGPOTrainer.build_advantages`.

## TODO 1: Signed Attention Transformer

Use the existing TurnRD idea: transformer over turn embeddings, score each turn, softmax over the real turns.

Problem:

```text
alpha_t = softmax(score_t)
alpha_t >= 0
sum_t alpha_t = 1
r_t = alpha_t * R
```

If `R >= 0`, every raw `r_t` is nonnegative. Group normalization can still make some turns negative relative to other trajectories at the same position, but inside one successful trajectory this does not directly say "this turn was bad."

Fix option A, centered softmax credit:

```text
r_t = s_i * (T_i * alpha_t - 1)
```

where:

```text
s_i = 2R_i - 1
```

or:

```text
s_i = R_i - mean_group(R)
```

Why this works:

- If `alpha_t > 1/T_i`, then turn `t` gets positive credit.
- If `alpha_t < 1/T_i`, then turn `t` gets negative credit.
- The output is length-normalized because uniform attention maps to zero.
- This avoids the "all positive turns" failure.

Important:

```text
sum_t r_t = 0
```

So turn rewards no longer decompose the final return. Set `lambda_consistency = 0.0`, or redefine the consistency loss for signed centered credit.

Fix option B, two-head signed attention:

```text
alpha^+_t = softmax(score^+_t)
alpha^-_t = softmax(score^-_t)
r_t = c^+_i alpha^+_t - c^-_i alpha^-_t
```

Useful budgets:

```text
c^+_i = R_i
c^-_i = 1 - R_i
```

or:

```text
c^+_i = max(R_i - baseline, 0)
c^-_i = max(baseline - R_i, 0)
```

This lets successful trajectories assign positive credit and failed trajectories assign negative credit.

Fix option C, signed value head:

```text
v_t = value_head(h_t)
r_t = v_t - mean_t(v_t)
```

Optionally gate by attention:

```text
r_t = alpha_t * (v_t - mean_t(v_t))
```

This is the cleanest way to let the model say a turn is actively bad.

Implementation TODO:

- Create `src/algorithms/hgpo/decomposers/turnrd_signed.py`.
- Reuse the TurnRDv2 encoder and add one of the signed reward heads above.
- Add config:

```json
"hgpo": {
  "alpha": 0.5,
  "lambda_consistency": 0.0,
  "decomposer": "turnrd_signed"
}
```

- Log:
  - `mean_abs_adv_token`;
  - signed reward mean/std;
  - fraction of negative turn rewards;
  - attention entropy;
  - correlation with raw env progress.

## TODO 2: Progress-Delta Reward

Use ALFWorld state changes as a cheap decomposer.

Candidate signals:

- `raw_env_reward` if populated by the adapter;
- whether the command was accepted;
- inventory changed;
- object moved to goal receptacle;
- object cleaned/heated/cooled;
- opened a useful receptacle;
- repeated action or no-op penalty.

Reward shape:

```text
r_t =
  w_valid * accepted_t
+ w_state * state_delta_t
+ w_repeat * repeat_penalty_t
+ w_terminal * terminal_bonus_t
```

TODO:

- Add `src/algorithms/hgpo/decomposers/alfworld_progress.py`.
- Keep it deterministic and parameter-free.
- Make this the first baseline for signed turn-level credit because it is debuggable.

## TODO 3: Counterfactual Delta Reward

Use the existing counterfactual machinery as the causal baseline:

```text
r_t = R(original) - mean_m R(replace action_t with alt_m and roll out)
```

TODO:

- Keep `n_turns_per_traj=0` for all turns.
- Never emit zeros for unsampled turns; either evaluate all turns or return only real labels through a separate mask-aware path.
- Cache by `(task_id, trajectory_id, turn_idx, action_text, policy_ckpt_id)`.
- Run only on short ALFWorld rollouts first because cost is high.

## TODO 4: Admissible-Action Margin Reward

For each turn with an oracle/expert action and admissible commands:

```text
margin_t = log p(expert_action) - max_{a in valid, a != expert} log p(a)
```

Turn reward:

```text
r_t = normalize(margin_t)
```

This is not causal by itself, but it is good for detecting whether the SFT policy knows the right action locally.

TODO:

- Add a cached candidate-scoring helper.
- Use this primarily for SFT diagnostics and as a weak auxiliary reward.
- Do not over-trust it during RL because the expert action may be unavailable after the policy drifts into a different state.

## Final Alpha Blend

Do not use a mixture of turn reward methods. Compare the individual
turn-level methods directly, then keep the proposal blend at:

```text
A_H = alpha * A_turn + (1 - alpha) * A_traj
```

with `alpha` sweep:

```text
alpha in {0.25, 0.5, 0.75}
```

## Variable-Length Trajectory/Padding Complications

These are the bugs to guard against.

1. Position-wise normalization compares absolute turn indices.

```text
A_turn(i,t) = normalize across trajectories that have turn t
```

For ALFWorld, turn 5 may mean "still searching" in one trajectory and "placing final object" in another. Absolute position can mix different semantic phases.

TODO: try phase bins:

```text
phase_t = t / (T_i - 1)
```

and normalize within phase buckets instead of raw `t`.

2. Late turns have fewer samples.

If only one trajectory reaches a late turn, then `K_t=1`, std is floored, and the advantage is effectively zero. Late recovery actions may get no signal.

TODO: add minimum-count diagnostics per position and log the fraction of turn positions with `K_t < 2`.

3. Softmax length dilution.

Uniform attention is `1/T_i`; longer trajectories get smaller per-turn raw rewards under `alpha_t * R_i`.

TODO: use centered credit:

```text
T_i * alpha_t - 1
```

or normalize reward magnitudes per trajectory before group normalization.

4. Padded zeros are poison.

If a decomposer returns zero for missing or unevaluated turns, `compute_turn_advantages` treats that zero as a real reward and creates fake negative examples.

TODO: absent turn means absent list entry, not `0.0`.

5. Long trajectories dominate token-mean PPO.

The current PPO loss averages over action tokens. A 20-turn trajectory contributes more terms than a 5-turn trajectory.

TODO: test a trajectory-balanced reduction:

```text
L = mean_i mean_t mean_u L_{i,t,u}
```

instead of one flat token mean.

6. Equal final rewards can still have turn signal.

This was patched: equal final rewards no longer force a skip if `combined` turn advantages are nonzero.

TODO: add a regression test once local Torch is healthy.

## Acceptance Checklist

Before a reward mechanism is "finalized":

- It returns `[K][T_i]` with no padding.
- It produces both positive and negative `A_turn` on a hand-built ALFWorld-like group.
- It has `lambda_consistency=0.0` unless the rewards truly sum to final return.
- It logs reward distribution and negative fraction.
- It has a smoke config under `maxrodriguez/configs/`.
- It can run 5 ALFWorld train episodes from an SFT adapter.
- It can evaluate both `valid_seen` and `valid_unseen` after the run.
