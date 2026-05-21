# Max Rodriguez ALFWorld Milestone Status Report

Date: 2026-05-19  
Workspace: `c:\Users\maxlr\CS224R`  
Primary code paths:

- [c:\Users\maxlr\CS224R\maxrodriguez\supervised_FT\app_alfworld_sft_plus.py](c:/Users/maxlr/CS224R/maxrodriguez/supervised_FT/app_alfworld_sft_plus.py)
- [c:\Users\maxlr\CS224R\maxrodriguez\grpo\app_alfworld_grpo.py](c:/Users/maxlr/CS224R/maxrodriguez/grpo/app_alfworld_grpo.py)
- [c:\Users\maxlr\CS224R\maxrodriguez\grpo\turn_level_reward_methods_todo.py](c:/Users/maxlr/CS224R/maxrodriguez/grpo/turn_level_reward_methods_todo.py)
- [c:\Users\maxlr\CS224R\maxrodriguez\grpo\app_signed_attention_transformer.py](c:/Users/maxlr/CS224R/maxrodriguez/grpo/app_signed_attention_transformer.py)
- [c:\Users\maxlr\CS224R\src\trainers\train_hgpo.py](c:/Users/maxlr/CS224R/src/trainers/train_hgpo.py)

## 1. Executive Summary

This workstream is building an ALFWorld behavior-cloning plus GRPO pipeline with a novel turn-level credit assignment proposal:

\[
A_H^{(i,t)} = \alpha A_{\text{turn}}^{(i,t)} + (1-\alpha) A_{\text{traj}}^{(i)}
\]

where:

- \(i\) indexes a sampled trajectory inside a GRPO group,
- \(t\) indexes a turn inside that trajectory,
- \(A_{\text{traj}}^{(i)}\) is the standard GRPO trajectory-level normalized advantage,
- \(A_{\text{turn}}^{(i,t)}\) is a turn-level normalized advantage from a decomposition method,
- \(\alpha \in [0,1]\) controls how much turn-level credit assignment influences the update.

The main milestone logic is:

1. build a strong SFT baseline,
2. evaluate it on full `valid_seen` and `valid_unseen`,
3. use the best SFT checkpoint as the single warm start for GRPO,
4. compare standard GRPO against turn-level reward variants.

Current headline result:

- final full SFT checkpoint scored:
  - `24 / 140 = 17.14%` on `valid_seen`
  - `21 / 134 = 15.67%` on `valid_unseen`
  - overall `45 / 274 = 16.42%`

The signed-attention transformer is now correctly wired into the live GRPO path. Old signed-attention GRPO runs that used the heuristic fallback have been stopped and should be disregarded.

---

## 2. ALFWorld Setup and Data

Official ALFWorld split sizes used as the target benchmark:

| Split | Count |
|---|---:|
| `train` | 3553 |
| `valid_seen` | 140 |
| `valid_unseen` | 134 |

Task categories:

| Task type | train | seen | unseen |
|---|---:|---:|---:|
| Pick & Place | 790 | 35 | 24 |
| Examine in Light | 308 | 13 | 18 |
| Clean & Place | 650 | 27 | 31 |
| Heat & Place | 459 | 16 | 23 |
| Cool & Place | 533 | 25 | 21 |
| Pick Two & Place | 813 | 24 | 17 |

### Expert SFT Data Generation

The SFT data file used in the milestone sweep is named:

`/vol/data/alfworld/sft_trajs_maxrodriguez_500.jsonl`

Despite the filename, the actual generated dataset contains:

- `6497` step-level SFT examples
- from `406` successful expert trajectories
- with the standard train/validation split:
  - `5977` train examples
  - `520` validation examples

This is documented in [c:\Users\maxlr\CS224R\maxrodriguez\docs\ALFWORLD_SFT_EXPERIMENTS.md](c:/Users/maxlr/CS224R/maxrodriguez/docs/ALFWORLD_SFT_EXPERIMENTS.md).

### Seen vs Unseen

- `valid_seen` means in-distribution generalization: same broad task families and room distributions as training, but new instances and layouts.
- `valid_unseen` means out-of-distribution generalization: unseen rooms and scene layouts, which is the harder benchmark.

---

## 3. Exact SFT Process Carried Out

### 3.1 Policy format

The SFT policy is a free-form ReAct policy over ALFWorld text prompts. The model sees a text prompt describing the state and recent interaction history, and it learns to generate the next action in text form.

At a high level, each training example is:

\[
(\text{prompt}, \text{target action text})
\]

where the target text is:

\[
\text{synthesize\_sft\_target(action)} + \text{eos}
\]

### 3.2 Tokenization and truncation

Each row is tokenized into:

- prompt tokens
- target action tokens

The prompt is truncated from the left first if the full example would exceed `max_seq_len`, so the action labels are preserved.

If

\[
|\text{prompt}| + |\text{target}| > L_{\max}
\]

then:

- if `target` itself is too long, keep only the last `L_max` target tokens,
- otherwise keep the last

\[
L_{\max} - |\text{target}|
\]

prompt tokens.

### 3.3 Loss

This is standard token-level behavior cloning with prompt masking.

Let:

- \(x = [x_1, \dots, x_n]\) be the full prompt-plus-target sequence,
- \(M_t = 1\) if token \(t\) is part of the action target,
- \(M_t = 0\) if token \(t\) is part of the prompt.

Then the loss is:

\[
\mathcal{L}_{\text{SFT}} =
- \frac{1}{\sum_t M_t}
  \sum_{t=1}^{n} M_t \log \pi_\theta(x_t \mid x_{<t})
\]

So the model is trained only on action tokens, not prompt tokens.

### 3.4 Practical training setup

The training loop uses:

- full finetuning
- `micro_batch_size = 1`
- gradient accumulation
- left-truncated prompt preservation
- optional length bucketing
- validation per-token CE after each epoch
- an admissible-action diagnostic on held-out validation examples

The exact script is:

[c:\Users\maxlr\CS224R\maxrodriguez\supervised_FT\app_alfworld_sft_plus.py](c:/Users/maxlr/CS224R/maxrodriguez/supervised_FT/app_alfworld_sft_plus.py)

---

## 4. Exact SFT Grid Search

### 4.1 Search space

The milestone SFT selection grid was:

| Hyperparameter | Values |
|---|---|
| epochs | `1` |
| learning rate | `1e-4`, `2e-5`, `1e-5`, `1e-6` |
| max sequence length | `1024`, `2048` |
| micro batch size | `1` |
| gradient accumulation | `4`, `8` |
| min reward filter | `1.0` |
| validation fraction | `0.08` |
| seed | `42` |
| use DAgger | `False` |

This produced `16` SFT runs.

### 4.2 Selection metric

The SFT sweep winner was selected by **validation per-token cross-entropy**, not by environment success rate.

Tracked metrics in each run:

- `train_per_token_ce`
- `val.per_token_ce`
- admissible-action diagnostic:
  - `admissible_top1`
  - `mean_expert_margin`

### 4.3 Full SFT grid leaderboard

Parsed from the `milestone_v8` submit logs:

| Rank | LR | Max Seq Len | Grad Accum | Validation CE |
|---:|---:|---:|---:|---:|
| 1 | `2e-5` | `1024` | `4` | `0.040079` |
| 2 | `2e-5` | `2048` | `8` | `0.042055` |
| 3 | `2e-5` | `2048` | `4` | `0.043002` |
| 4 | `2e-5` | `1024` | `8` | `0.043681` |
| 5 | `1e-5` | `1024` | `4` | `0.054074` |
| 6 | `1e-5` | `2048` | `8` | `0.054997` |
| 7 | `1e-5` | `1024` | `8` | `0.055886` |
| 8 | `1e-5` | `2048` | `4` | `0.058257` |
| 9 | `1e-4` | `2048` | `4` | `0.080298` |
| 10 | `1e-4` | `1024` | `4` | `0.081833` |
| 11 | `1e-4` | `2048` | `8` | `0.086655` |
| 12 | `1e-4` | `1024` | `8` | `0.094192` |
| 13 | `1e-6` | `1024` | `8` | `0.313496` |
| 14 | `1e-6` | `2048` | `8` | `0.313797` |
| 15 | `1e-6` | `1024` | `4` | `0.319481` |
| 16 | `1e-6` | `2048` | `4` | `0.320460` |

Winner note saved at:

[c:\Users\maxlr\CS224R\maxrodriguez\results\sft_grid_winner_note.txt](c:/Users/maxlr/CS224R/maxrodriguez/results/sft_grid_winner_note.txt)

Winner:

- `learning_rate = 2e-5`
- `max_seq_len = 1024`
- `grad_accum = 4`

---

## 5. Final SFT Training Run

### 5.1 Exact final training configuration

The winning SFT configuration was then rerun for `3` epochs:

- learning rate: `2e-5`
- max sequence length: `1024`
- micro batch size: `1`
- grad accumulation: `4`
- seed: `42`
- DAgger: `False`

Final checkpoint:

`/vol/checkpoints/maxrodriguez_milestone/final_sft/sftfinal_milestone_v8_e3_lr2E-05_seq1024_ga4_nodagger`

### 5.2 Epoch-by-epoch final SFT metrics

Parsed from the final SFT training log:

| Epoch | Train Token CE | Validation Token CE | Admissible Top-1 | Mean Expert Margin |
|---:|---:|---:|---:|---:|
| 0 | `0.083691` | `0.041164` | `0.890625` | `0.377942` |
| 1 | `0.037811` | `0.037651` | `0.90625` | `0.710135` |
| 2 | `0.027234` | `0.037339` | `0.890625` | `0.671301` |

Interpretation:

- epoch `1` gave a large jump in both CE and admissible-action margin,
- epoch `2` slightly improved CE further,
- the run remained stable,
- no DAgger data was injected.

---

## 6. Final Full SFT Evaluation

### 6.1 Exact evaluation setup

The final SFT checkpoint was evaluated on the full benchmark:

- `140` seen tasks
- `134` unseen tasks

This is the correct final benchmark protocol, not the earlier small smoke-test subsets.

### 6.2 Results

| Split | Successes | Total | Success Rate |
|---|---:|---:|---:|
| `valid_seen` | `24` | `140` | `17.14%` |
| `valid_unseen` | `21` | `134` | `15.67%` |
| combined | `45` | `274` | `16.42%` |

These values come from:

- [c:\Users\maxlr\AppData\Local\CodexModalMilestone\milestone_v8\submit_logs\sfteval_milestone_v8_seen140.out.log](c:/Users/maxlr/AppData/Local/CodexModalMilestone/milestone_v8/submit_logs/sfteval_milestone_v8_seen140.out.log)
- [c:\Users\maxlr\AppData\Local\CodexModalMilestone\milestone_v8\submit_logs\sfteval_milestone_v8_unseen134.out.log](c:/Users/maxlr/AppData/Local/CodexModalMilestone/milestone_v8/submit_logs/sfteval_milestone_v8_unseen134.out.log)

### 6.3 Interpretation

This final SFT result is the current best clean milestone baseline in the Max pipeline:

- it is trained with the selected hyperparameters from a real sweep,
- it is evaluated on the full official seen/unseen splits,
- it serves as the single warm-start checkpoint for GRPO experiments.

---

## 7. Historical Exploratory SFT Results

Before the current milestone pipeline, several exploratory SFT runs were documented in:

[c:\Users\maxlr\CS224R\maxrodriguez\docs\ALFWORLD_SFT_EXPERIMENTS.md](c:/Users/maxlr/CS224R/maxrodriguez/docs/ALFWORLD_SFT_EXPERIMENTS.md)

Key historical results:

| Method | Checkpoint | Eval | Result |
|---|---|---|---:|
| Out-of-box Qwen2.5-1.5B | none | rerank 20 | `0 / 20 = 0%` |
| 3-epoch LoRA SFT | `maxrodriguez_sft500_r32_len2048_e3_20260514_024413` | rerank OOD 20 | `6 / 20 = 30%` |
| 3-epoch LoRA SFT | same | free-form OOD 50 | `14 / 50 = 28%` |
| Best 1-epoch LoRA sweep | `lora_r32_lr1e4_do05_e1` | free-form OOD 20 | `4 / 20 = 20%` |
| Best 1-epoch full FT sweep | `full_lr1e5_e1` | free-form OOD 20 | `5 / 20 = 25%` |

These are useful context, but the current milestone path should prioritize the newer full-benchmark final SFT numbers above.

---

## 8. Exact GRPO Proposal and Math

### 8.1 Standard trajectory-level GRPO

Suppose we sample a GRPO group of \(K\) trajectories for the same task prompt, with final rewards

\[
R_1, R_2, \dots, R_K
\]

Then the trajectory-level normalized GRPO advantage is:

\[
A_{\text{traj}}^{(i)} =
\frac{R_i - \mu_R}{\sigma_R + \varepsilon}
\]

where

\[
\mu_R = \frac{1}{K}\sum_{j=1}^{K} R_j
\qquad
\sigma_R = \sqrt{\frac{1}{K}\sum_{j=1}^{K}(R_j - \mu_R)^2}
\]

In plain language: better-than-group trajectories get positive advantage, worse-than-group trajectories get negative advantage.

### 8.2 Proposed turn-level variant

Each trajectory \(i\) has turns

\[
t = 1, 2, \dots, T_i
\]

For some turn-level reward decomposition method, we first produce raw turn rewards

\[
r_{\text{turn}}^{(i,1)}, \dots, r_{\text{turn}}^{(i,T_i)}
\]

Then we normalize by turn index across the GRPO group:

\[
A_{\text{turn}}^{(i,t)} =
\frac{r_{\text{turn}}^{(i,t)} - \mu_t}{\sigma_t + \varepsilon}
\]

where \(\mu_t\) and \(\sigma_t\) are computed only over trajectories that actually have a real turn \(t\). This avoids padding bugs from different trajectory lengths.

The final hybrid advantage is:

\[
A_H^{(i,t)} = \alpha A_{\text{turn}}^{(i,t)} + (1-\alpha)A_{\text{traj}}^{(i)}
\]

This means:

- if `alpha = 0`, the method reduces to standard GRPO,
- if `alpha = 1`, the update is fully turn-level,
- intermediate values blend sparse outcome credit with local turn credit.

### 8.3 Policy loss

For action token \(u\) inside turn \(t\) of trajectory \(i\), let:

- \(\log \pi_\theta\) be the current policy logprob,
- \(\log \pi_{\text{old}}\) be the behavior policy logprob,
- \(A_H^{(i,t)}\) be copied onto all action tokens in that turn.

Then the PPO-style ratio is:

\[
\rho = \exp(\log \pi_\theta - \log \pi_{\text{old}})
\]

The clipped objective is:

\[
\mathcal{L}_{\text{policy}} =
-\mathbb{E}\left[
\min\left(
\rho A_H,
\text{clip}(\rho, 1-\epsilon, 1+\epsilon)A_H
\right)
\right]
\]

with an optional KL penalty against a reference model:

\[
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{policy}} + \beta_{\text{KL}}\mathcal{L}_{\text{KL}}
\]

This is the exact place where “good turns positive, bad turns negative” actually enters the gradient.

---

## 9. Exact Turn-Level Reward Variants

### 9.1 Standard GRPO baseline: `trajectory_only`

Motivation:

- establish the baseline with no turn-level shaping.

Definition:

\[
r_{\text{turn}}^{(i,t)} = 0
\]

for every turn, so only \(A_{\text{traj}}^{(i)}\) contributes.

### 9.2 `progress_delta`

Motivation:

- ALFWorld exposes dense intermediate progress signals,
- this is the simplest direct turn reward.

Definition:

For each turn:

\[
r_{\text{turn}}^{(i,t)} =
\text{intermediate\_reward}^{(i,t)}
\]

and if the whole episode succeeds, add a terminal bonus on the last turn:

\[
r_{\text{turn}}^{(i,T_i)}
\leftarrow
r_{\text{turn}}^{(i,T_i)} + b_{\text{terminal}}
\]

Why it matters:

- it gives immediate positive or negative credit to progress-making turns,
- it is the most trustworthy current turn-level variant.

### 9.3 `signed_attention`

Motivation:

- not every turn should contribute equally,
- some turns are bottlenecks,
- a learned attention-style scorer can concentrate credit on the important turns.

#### Old problem

A raw softmax produces only positive weights:

\[
w_t = \text{softmax}(s_t)
\]

That means every turn gets nonnegative mass, which is a problem if we want some turns to be actively penalized.

#### Fix: centered attention

The transformer produces scores \(s_t\), then:

\[
w_t = \text{softmax}(s_t)
\]

and then the weights are centered:

\[
\tilde{w}_t = T_i \cdot w_t - 1
\]

This guarantees:

\[
\frac{1}{T_i}\sum_{t=1}^{T_i} \tilde{w}_t = 0
\]

So:

- some turns can be positive,
- some turns can be negative,
- the average turn credit is zero before the outcome sign is applied.

#### Transformer target

The signed-attention transformer is trained on centered ALFWorld progress targets:

\[
\text{progress}_t = \text{intermediate\_reward}_t
\]

\[
\bar{p}_t = p_t - \frac{1}{T_i}\sum_{\tau=1}^{T_i} p_\tau
\]

\[
\hat{p}_t = \frac{\bar{p}_t}{\max_\tau |\bar{p}_\tau| + \varepsilon}
\]

\[
y_t = \text{sign}(R_i)\hat{p}_t
\]

and the transformer minimizes:

\[
\mathcal{L}_{\text{SA}} =
\frac{1}{T_i}\sum_{t=1}^{T_i}(\tilde{w}_t - y_t)^2
\]

This is now wired into the live GRPO path. A `signed_attention` GRPO run must provide a trained transformer checkpoint, otherwise it fails fast.

### 9.4 `admissible_margin`

Motivation:

- if the chosen action is much better than alternatives under the policy, that may be useful local credit.

Definition:

If the chosen action logprob is \( \ell_{\text{chosen}} \) and the best admissible alternative is \( \ell_{\text{alt}}^\star \), define:

\[
r_{\text{turn}}^{(i,t)} = \ell_{\text{chosen}} - \ell_{\text{alt}}^\star
\]

Optionally normalize it.

Current caveat:

- the rollout path still does not robustly populate admissible alternative logprobs,
- so this variant is currently more speculative and may be weak until that data path is improved.

### 9.5 `counterfactual_delta` (deferred)

Motivation:

- estimate the causal effect of a chosen turn by comparing the realized return to returns from counterfactual alternative actions.

High-level definition:

\[
r_{\text{turn}}^{(i,t)} =
R_i - \frac{1}{M}\sum_{m=1}^{M} R_{\text{cf}}^{(i,t,m)}
\]

where \(R_{\text{cf}}^{(i,t,m)}\) is the return after replacing turn \(t\) with alternative action \(m\) and rolling out the rest.

This is the most expensive and most causally direct variant, but it is explicitly deferred until after the milestone.

---

## 10. Exact GRPO Grid(s)

### 10.1 Original milestone GRPO grid

The milestone GRPO search space in the Max folder is:

| Hyperparameter | Values |
|---|---|
| method | `trajectory_only`, `progress_delta`, `signed_attention`, `admissible_margin` |
| alpha | baseline `0.0`; variants `0.1`, `0.5`, `0.9` |
| learning rate | `1e-6`, `2e-6`, `5e-6` |
| KL coeff | `0.02`, `0.05` |
| clip epsilon | `0.2` |
| K trajectories per task | `4` |
| grad accumulation | `1` |
| max tokens per microbatch | `2048` |
| KL warmup episodes | `5` |
| episodes | `100` |
| max turns | `30` |

For cheap grid search, the environment caps were:

- train pool: `200` train games
- inline eval pool: `50`

### 10.2 Current corrected scheduling

Because signed attention now requires a trained transformer, the GRPO plan is now split:

1. run the **core GRPO sweep**:
   - `trajectory_only`
   - `progress_delta`
   - `admissible_margin`

2. run a **signed-attention transformer grid**

3. train the **best final signed-attention transformer**

4. run a **signed-attention-only GRPO sweep** using that trained transformer checkpoint

5. run final GRPO confirmation training for the winning variant

This is now supported by:

- [c:\Users\maxlr\CS224R\maxrodriguez\scripts\run_milestone_gridsearch.ps1](c:/Users/maxlr/CS224R/maxrodriguez/scripts/run_milestone_gridsearch.ps1)
- [c:\Users\maxlr\CS224R\maxrodriguez\tools\launch_grpo_grid.py](c:/Users/maxlr/CS224R/maxrodriguez/tools/launch_grpo_grid.py)

---

## 11. Signed-Attention Transformer Status

### 11.1 What was wrong before

Earlier `signed_attention` GRPO runs were not actually using a trained transformer. They were falling back to a heuristic scorer because no model checkpoint was being loaded.

### 11.2 What is fixed now

The live GRPO branch now:

- loads a trained signed-attention transformer checkpoint,
- passes it into the decomposer,
- refuses to run `signed_attention` without that checkpoint.

### 11.3 Signed-attention transformer grid

Current transformer grid:

| Hyperparameter | Values |
|---|---|
| epochs | `1` |
| learning rate | `5e-5`, `1e-4`, `2e-4` |
| hidden size | `64`, `128` |
| transformer layers | `1`, `2` |
| attention heads | `4` |
| dropout | `0.0` |
| train trajectories | `256` |
| val trajectories | `64` |
| max turns | `30` |
| seed | `42` |

Final transformer fit:

- same best hyperparameters
- `3` epochs

### 11.4 Current status

- transformer training app exists,
- transformer checkpoint save/load is implemented,
- live GRPO wiring is fixed,
- old signed-attention GRPO launches have been stopped,
- no valid signed-attention GRPO result should be reported until the transformer grid and final transformer fit are completed.

---

## 12. Current GRPO Status

### 12.1 What has been tried

Multiple detached GRPO launch attempts were made for:

- `trajectory_only`
- `progress_delta`
- `signed_attention`
- `admissible_margin`

### 12.2 Why current GRPO results should not be trusted yet

There were several invalid or incomplete launch attempts:

1. an early launch path failed due Modal secret handling,
2. some runs later showed ALFWorld import/runtime crashes,
3. signed-attention runs before the recent fix were not using a trained transformer,
4. stale background GRPO launchers were stopped and those runs should be disregarded.

### 12.3 Current bottom line

As of this report:

- **there are no valid final GRPO baseline or variant benchmark results to report yet**,
- **old signed-attention GRPO runs should be discarded**,
- the pipeline is now correctly structured for:
  - core GRPO sweep,
  - signed-attention transformer sweep,
  - signed-attention final transformer fit,
  - signed-attention GRPO sweep,
  - final GRPO confirmation training.

---

## 13. Exact Comparisons Being Made

### 13.1 SFT comparisons

Current completed comparison:

- 16-way 1-epoch SFT hyperparameter sweep, selected by validation CE

Current final benchmark comparison:

- best 3-epoch no-DAgger SFT on full `valid_seen` and `valid_unseen`

Planned next SFT comparison:

- `no dagger` vs `minimal dagger`

### 13.2 RL comparisons

Planned GRPO comparison matrix:

| Family | Method | Purpose |
|---|---|---|
| baseline | `trajectory_only` | standard GRPO |
| turn-level | `progress_delta` | dense environment progress |
| turn-level | `admissible_margin` | policy confidence gap |
| turn-level | `signed_attention` | learned centered transformer turn credit |
| post-milestone | `counterfactual_delta` | causal turn credit via replay |

Each turn-level method will be compared against standard GRPO under the same SFT warm start and with its own optimized GRPO hyperparameters.

---

## 14. What Has Been Documented / Saved

Local milestone winner note:

- [c:\Users\maxlr\CS224R\maxrodriguez\results\sft_grid_winner_note.txt](c:/Users/maxlr/CS224R/maxrodriguez/results/sft_grid_winner_note.txt)

Launch ledger:

- [c:\Users\maxlr\CS224R\maxrodriguez\results\milestone_launch_ledger.jsonl](c:/Users/maxlr/CS224R/maxrodriguez/results/milestone_launch_ledger.jsonl)

Final SFT eval logs:

- [c:\Users\maxlr\AppData\Local\CodexModalMilestone\milestone_v8\submit_logs\sfteval_milestone_v8_seen140.out.log](c:/Users/maxlr/AppData/Local/CodexModalMilestone/milestone_v8/submit_logs/sfteval_milestone_v8_seen140.out.log)
- [c:\Users\maxlr\AppData\Local\CodexModalMilestone\milestone_v8\submit_logs\sfteval_milestone_v8_unseen134.out.log](c:/Users/maxlr/AppData/Local/CodexModalMilestone/milestone_v8/submit_logs/sfteval_milestone_v8_unseen134.out.log)

Final SFT training log:

- [c:\Users\maxlr\AppData\Local\CodexModalMilestone\milestone_v8\submit_logs\sftfinal_milestone_v8_e3_lr2E-05_seq1024_ga4_nodagger.out.log](c:/Users/maxlr/AppData/Local/CodexModalMilestone/milestone_v8/submit_logs/sftfinal_milestone_v8_e3_lr2E-05_seq1024_ga4_nodagger.out.log)

---

## 15. Future Work for Final Report

The following should be explicitly included in the final project report as remaining work or next-step experiments:

1. **Add DAgger to SFT**
   - run a clean `no dagger` vs `minimal dagger` comparison
   - use the same winning SFT hyperparameters for both

2. **Choose best hyperparameters based on actual environment success, not only CE**
   - the current SFT sweep chose the winner by validation token CE
   - for the final project, hyperparameter selection should include actual `valid_seen` and `valid_unseen` success rates

3. **Finetune the signed-attention transformer**
   - run the signed-attention transformer grid
   - train the best final transformer for longer
   - then rerun the signed-attention GRPO variant with the trained transformer checkpoint

4. **Try the counterfactual variant**
   - this is post-milestone because it requires replay, counterfactual rollouts, and caching
   - but it is one of the most important remaining variants conceptually

5. **Produce final GRPO comparison tables**
   - standard GRPO vs `progress_delta`
   - standard GRPO vs `admissible_margin`
   - standard GRPO vs `signed_attention`
   - and eventually standard GRPO vs `counterfactual_delta`

6. **Prefer final model selection by full benchmark generalization**
   - full `140` seen and `134` unseen
   - not small smoke-test subsets
   - not pure validation CE alone

