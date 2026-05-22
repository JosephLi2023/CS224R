# Max Rodriguez ALFWorld Milestone Workspace

This folder is organized as the working area for the remaining ALFWorld
milestone work: checkpoint selection, turn-level reward implementation, smoke
tests, and final seen/unseen experiments.

## 0. Folder Map

```text
maxrodriguez/
  apps/          Modal apps for SFT data generation, SFT training, eval, DAgger
  configs/       Place future smoke/final configs here
  checkpoints/   Local checkpoint copies, grouped by source
  docs/          Experiment logs and benchmark runbooks
  scripts/       Local command wrappers
  todos/         Turn-level reward specs and implementation stubs
  tools/         Local utilities
  README.md      This milestone playbook
```

## 1. Checkpoint Status

### Best SFT Checkpoint To Use

Authoritative Modal path:

```text
/vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2
```

This is the checkpoint that solved a sanity task:

```text
eval_out_of_distribution
task_id=6502
success=True
turns=8
```

It was trained with:

```powershell
modal run --detach maxrodriguez/apps/app_alfworld_sft_plus.py::train_sft_plus `
  --epochs 10 `
  --learning-rate 0.00001 `
  --max-seq-len 2048 `
  --full-finetune True `
  --run-name overnight_full_lr1e5_e10 `
  --data-path /vol/data/alfworld/sft_trajs_maxrodriguez_500.jsonl `
  --output-dir /vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2
```

Training type:

```text
behavior cloning / supervised fine-tuning
masked cross-entropy on expert ReAct targets
not RL
not DAgger
```

Important pipeline note:

```text
infra/app_train_loop.py currently warm-starts RL by calling LoRAPolicy.load_adapter(...).
That path expects a PEFT/LoRA adapter, not a full-model checkpoint.
```

So the full checkpoint is immediately usable for `app_alfworld_sft_plus.py`
free-form evaluation. To use it as the RL starting point, do one of:

1. add full-checkpoint loading support to the RL train loop; or
2. train a LoRA adapter initialized from this full checkpoint and use that
   LoRA adapter for GRPO.

Do not accidentally pass this full checkpoint to `--sft-adapter` in the
current RL loop until that support is added.

### Local Checkpoint Copies

Do not keep a local copy of the best full SFT checkpoint. It is several GB and
should stay on the Modal volume.

Local path intentionally absent:

```text
checkpoints/best_full_sft_alfworld/
```

Use this Modal path whenever the best SFT checkpoint is needed:

```text
/vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2
```

### Google Drive Checkpoints

The Drive adapters are intentionally labeled by source and likely task:

```text
checkpoints/google_drive/alfworld_lora_sft_20260507_165617
checkpoints/google_drive/webshop_lora_sft_v3_20260504_154752
```

The failed ALFWorld eval used:

```text
checkpoints/google_drive/alfworld_lora_sft_20260507_165617
```

So the failure was not because we accidentally evaluated the WebShop adapter.
The other adapter, `webshop_lora_sft_v3_20260504_154752`, is likely the WebShop
SFT adapter and should not be used for ALFWorld.

Validate local LoRA adapters with:

```powershell
python maxrodriguez/tools/smoke_check_sft_checkpoint.py
```

## 2. Code Already Used For SFT And Eval

Main app:

```text
maxrodriguez/apps/app_alfworld_sft_plus.py
```

Important functions:

```text
train_sft_plus              # BC/SFT training
evaluate_freeform_greedy    # raw free-form ReAct eval
evaluate_action_rerank      # valid-action rerank eval
```

Experiment log:

```text
maxrodriguez/docs/ALFWORLD_SFT_EXPERIMENTS.md
```

Post-SFT runbook:

```text
maxrodriguez/docs/ALFWORLD_POST_SFT_BENCHMARK.md
```

## 3. Milestone Goal

By the milestone, we want:

1. a working SFT baseline checkpoint identified and sanity checked;
2. at least one new turn-level reward method implemented;
3. the method wired into the GRPO/HGPO decomposer pipeline;
4. smoke tests showing the reward shape and alpha blend are correct;
5. a small ALFWorld training smoke run;
6. final evaluation on both `valid_seen` and `valid_unseen`.

The proposal alpha semantics must remain:

```text
A_H(i,t) = alpha * A_turn(i,t) + (1 - alpha) * A_traj(i)
```

## 4. Turn-Level Reward Implementation TODOs

### Files To Edit First

Primary implementation stubs:

```text
maxrodriguez/todos/turn_level_reward_methods_todo.py
```

Detailed math/design spec:

```text
maxrodriguez/todos/TURN_LEVEL_REWARD_TODOS.md
```

Stub classes:

```text
SignedAttentionTransformerTODO
ALFWorldProgressDeltaTODO
CounterfactualDeltaTODO
AdmissibleActionMarginTODO
```

Every method must return:

```text
list[list[float]]
```

with ragged shape:

```text
[K][T_i]
```

Meaning:

```text
K      = number of trajectories in the GRPO group
T_i    = number of real turns in trajectory i
out[i] = one reward per real turn
```

No padding. No fake zeros for missing turns.

### Recommended First Implementation

Implement this first because it is easiest to debug:

```text
ALFWorldProgressDeltaTODO
```

Target reward:

```text
r_t =
  w_valid    * accepted_t
+ w_state    * state_delta_t
+ w_repeat   * repeat_penalty_t
+ w_terminal * terminal_bonus_t
```

Then implement:

```text
SignedAttentionTransformerTODO
```

Use the signed centered softmax fix:

```text
a_it = softmax(score_it over real turns)
s_i = 2R_i - 1
r_it = s_i * (T_i * a_it - 1)
```

This solves the vanilla-softmax issue:

```text
a_it >= 0
```

but:

```text
T_i * a_it - 1
```

can be positive or negative.

## 5. Local Tests To Add

Add focused tests before wiring into the training loop.

Suggested test file:

```text
tests/unit/test_maxrodriguez_turn_rewards.py
```

Minimum checks:

```text
1. output shape is exactly [K][T_i]
2. variable-length trajectories are handled
3. no NaN or inf
4. no padded missing turns
5. after compute_turn_advantages, at least one positive and one negative turn advantage exists
6. alpha blend still computes alpha * A_turn + (1 - alpha) * A_traj
```

Run:

```powershell
python -m pytest tests/unit/test_maxrodriguez_turn_rewards.py tests/unit/test_hgpo_advantage.py
```

## 6. Wire A Finished Method Into The Pipeline

Once a TODO method works locally, move or import it into:

```text
src/algorithms/hgpo/decomposers/
```

Example:

```text
src/algorithms/hgpo/decomposers/alfworld_progress.py
src/algorithms/hgpo/decomposers/turnrd_signed.py
```

Then add a branch in:

```text
src/algorithms/hgpo/decomposers/base.py::build_decomposer
```

Add a smoke config under:

```text
maxrodriguez/configs/
```

Config must include:

```json
{
  "hgpo": {
    "alpha": 0.5,
    "lambda_consistency": 0.0,
    "decomposer": "<your_method_name>"
  }
}
```

Use `lambda_consistency = 0.0` for signed centered rewards unless the turn
rewards truly sum to the final trajectory return.

## 7. RL Warm-Start Decision

Before launching RL, decide which checkpoint format the RL loop will use.

Current situation:

```text
Best performing SFT checkpoint: full model
Current RL warm-start path: LoRA adapter loader
```

Therefore choose one path:

### Path A: Add Full-Checkpoint Support

Modify the RL policy construction so it can load:

```text
/vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2
```

as the base model before attaching trainable LoRA for RL.

Expected touch points:

```text
src/policy/lora_policy.py
infra/app_train_loop.py
scripts/run_turnrd_modal.py
```

### Path B: Train A New LoRA Adapter From The Full Checkpoint

Use `train_sft_plus` with:

```text
--base-model-path /vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2
```

and `full_finetune=False`, then use the resulting LoRA adapter as
`--sft-adapter` for GRPO.

This is likely the fastest path if you want to avoid refactoring the RL loader.

## 8. Smoke Evaluation Commands

### Confirm Full Checkpoint Still Solves At Least One Task

```powershell
modal run maxrodriguez/apps/app_alfworld_sft_plus.py `
  --action freeform_eval `
  --adapter-path /vol/checkpoints/maxrodriguez_overnight/full_lr1e5_e10_v2 `
  --checkpoint-type full `
  --episodes 20 `
  --task-id-base 6500 `
  --max-turns 40 `
  --max-seq-len 2048 `
  --split eval_out_of_distribution `
  --run-name sanity_full_e10_stop_on_success `
  --stop-after-success
```

Expected sanity result:

```text
at least one success within the first few tasks
```

### Tiny RL Smoke After A LoRA-Compatible Warm Start Exists

Only run this after Path A or Path B above is resolved:

```powershell
python scripts/run_turnrd_modal.py `
  --env-name alfworld `
  --config maxrodriguez/configs/<your_method_smoke_config>.json `
  --rounds 1 `
  --episodes-per-round 5 `
  --sft-adapter /vol/checkpoints/<lora_compatible_sft_adapter> `
  --carry-policy-across-rounds `
  --eval-episodes 5 `
  --eval-task-id-base 6500
```

Check logs for:

```text
mean_abs_adv_token > 0
dead_K_group = 0 when turn signal exists
nonzero mean_turn_adv
no shape errors
no padded-turn errors
```

## 9. Final Milestone Experiments

Once smoke passes, run:

```text
alpha in {0.25, 0.50, 0.75}
```

For each alpha:

```text
rounds = 5
episodes_per_round = 40
K = 4
max_turns = 30 or 40, but keep it fixed across methods
```

Evaluate each final adapter on:

```text
valid_seen   = eval_in_distribution, 140 episodes
valid_unseen = eval_out_of_distribution, 134 episodes
```

Use:

```powershell
modal run maxrodriguez/apps/app_alfworld_sft_plus.py `
  --action freeform_eval `
  --adapter-path /vol/checkpoints/<final_adapter> `
  --checkpoint-type lora `
  --episodes 140 `
  --task-id-base 0 `
  --split eval_in_distribution `
  --run-name <method>_seen140_freeform

modal run maxrodriguez/apps/app_alfworld_sft_plus.py `
  --action freeform_eval `
  --adapter-path /vol/checkpoints/<final_adapter> `
  --checkpoint-type lora `
  --episodes 134 `
  --task-id-base 0 `
  --split eval_out_of_distribution `
  --run-name <method>_unseen134_freeform
```

Report:

```text
1. valid_seen success rate
2. valid_unseen success rate
3. average turns
4. alpha
5. reward method
6. whether reward outputs had positive and negative turn advantages
7. whether variable-length trajectories were present
8. final checkpoint path
```

## 10. Quick Checklist

Before calling the milestone done:

```text
[ ] README paths match current folder structure
[ ] local Google Drive adapters are labeled by source/task
[ ] best full SFT checkpoint path is documented
[ ] no partial local full checkpoint is being used
[ ] one turn reward TODO is implemented
[ ] unit tests pass
[ ] decomposer is wired into build_decomposer
[ ] smoke config exists in maxrodriguez/configs
[ ] tiny ALFWorld RL smoke run completes
[ ] final seen/unseen evals complete
[ ] final numbers are copied into docs/ALFWORLD_POST_SFT_BENCHMARK.md
```
