# CS224R H-GRPO — Execution Plan

## Context

We are implementing **H-GRPO** (Hierarchical Group Relative Policy Optimization for multi-turn LLM agents) per the proposal at `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/CS224R_Project_Proposal.pdf`. Key deadlines:

- **Milestone (1 page + 1 experiment): 5/22 9pm** — code-complete flat GRPO baseline + H-GRPO Method A (open-source LLM judge) + H-GRPO Method B (TurnRD), with initial WebShop results.
- **Poster: 6/3 9am submission, presentation 9:30am–1:45pm** — add ALFWorld, ablations (α, consistency reg, Method A/B/C, TurnRD refresh cadence), gradient-variance analysis.
- **Final report: 6/8 9pm** — 8-page report + 1-page extended abstract.

### Why this plan exists

The current scaffold at `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/` (built by Codex) provides good directory structure, env-adapter shells, and reproducibility infra — but the actual H-GRPO algorithm is **not** implemented:

- `src/algorithms/hgpo/objective.py:hgpo_action_bonus` treats "groups" as partitions of bandit-action indices (e.g., `{0:[0,1,2,3], 1:[4,5,6,7]}`) and adds `alpha * (group_mean − global_mean)` to action returns. The proposal's "groups" are K=4 sampled trajectories per task, normalized at trajectory and turn levels.
- No LLM (`requirements.txt` is empty), no LoRA, no vLLM, no PyTorch, no per-turn reward decomposition, no consistency regularizer, no Modal harness.
- `src/algorithms/baseline/policy.py:SoftmaxPolicy.sample_text_action` picks `valid_actions[idx % len(valid_actions)]` — i.e., not LLM-driven.

This plan rewrites the algorithm + training stack while preserving the layout, env adapters, IO/checkpoint utilities, evaluator, factory, and test scaffolding.

### Frozen design decisions (from user Q&A)

| Decision | Choice |
|---|---|
| RL stack | Lean custom: TRL GRPO primitives + PEFT/LoRA, custom multi-turn loop |
| Rollout inference | vLLM from day 1 |
| Judge model | **Qwen2.5-7B-Instruct** via vLLM |
| Compute | 1× A100-80GB (trainer + policy vLLM) + 1× A10G (judge vLLM) on Modal, shared Volume |
| Envs | Built fresh inside Modal image (WebShop + ALFWorld) |
| Milestone scope | Flat GRPO + Method A (Judge) + Method B (TurnRD) on WebShop, code-complete + initial result |
| Team | Joseph executes all coding; teammates review |

---

## Goal

1. Re-implement H-GRPO faithfully per Section 3.1 of the proposal:
   - K=4 trajectories per task from a Qwen2.5-1.5B + LoRA policy.
   - Trajectory-level group-normalized advantage `Â_traj(τ_i) = (R_i − R̄)/σ_R`.
   - Turn-level group-normalized advantage `Â_turn(t,τ_i) = (r̂_t^i − r̄_t)/σ_{r̂_t}` from per-turn reward decomposition.
   - Combined per-turn advantage `Â_H = α·Â_traj + (1−α)·Â_turn` (α=1 must reduce exactly to flat GRPO).
   - Consistency regularizer `λ·‖Σ_t Â_turn(t,τ_i) − Â_traj(τ_i)‖²`.
2. Implement three reward decomposers (A: judge, B: TurnRD, C: progress).
3. Run on WebShop (milestone) and ALFWorld (poster), with a frozen ReAct baseline + flat GRPO + ArCHer-style critic baseline.
4. Produce milestone, poster, final report on schedule.

---

## Approach

### Architectural shape

- **Custom `HGPOTrainer`** (not subclassed from `trl.GRPOTrainer`, since TRL's GRPO is single-turn and doesn't expose turn-level signals). Borrow infra patterns from `trl/trainer/grpo_trainer.py` (KL controller, ref-model logprob computation, optimizer plumbing) and the `verl-agent/recipe/hgpo` reference for advantage math + per-turn reward plumbing.
- **Policy = HuggingFace `Qwen2.5-1.5B-Instruct` with PEFT LoRA** (q,k,v,o_proj). Trainer holds LoRA params + base weights; reference model = frozen base (no LoRA).
- **Rollout = vLLM server** in the same process. After each optimizer step, push merged LoRA weights into vLLM (`vllm.LLM.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(...)` pattern). For Method B we additionally pull mean-pooled hidden states from a forward pass at logprob-recompute time (vLLM gives us tokens; we recompute hidden states on the trainer model in eval mode).
- **Judge = separate Modal app** (`app_judge.py`) serving Qwen2.5-7B-Instruct via vLLM behind an HTTP endpoint `/score_turns`. Trainer calls it asynchronously per trajectory batch with a prompt template asking for per-turn 0-10 scores given the full trajectory + final outcome. Scores normalized so `Σ_t r̂_t = R_traj`. Caching by `(task_id, turn_idx, prefix_hash)` in a Modal Volume (sqlite).
- **TurnRD** (Method B): small Transformer (4 layers, hidden 256, ~8M params) over per-turn embeddings + `[CLS]`. Three training modes per proposal Section 3.2; we implement Mode 1 (default) and Mode 2 (distill judge) for the milestone, defer Mode 3 (contrastive) to stretch goal. Refreshed every N=20 episodes on a held-out trajectory replay buffer.
- **Modal layout**: one image baked with all deps + WebShop + ALFWorld data; three Modal apps (`train`, `judge`, `eval`) mounting the same Volume for checkpoints/cache/logs.

### What gets reused vs rewritten

**Keep as-is or with minor refactor:**

- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/factory.py` — extend with prompt-format hooks.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/webshop_adapter.py` and `alfworld_adapter.py` — already handle reset/step normalization and admissible-action extraction; will need (a) a `format_prompt(state)` method per env and (b) an `info["score"]` accessor for Method C progress signal.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/trainers/io_utils.py` — keep `dump_json/load_json/save_checkpoint/load_checkpoint/deep_merge/ensure_dir`; extend `save_checkpoint` to handle PEFT adapter state + optimizer state.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/trainers/evaluator.py` — refactor `evaluate_policy` to take an `LLMPolicy` instead of `SoftmaxPolicy`; reuse `EvalResult` shape and add `success_rate`, `mean_reward`, `mean_length` fields.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/toy_bandit.py` — keep as deterministic regression target; we'll add a *toy multi-turn* env for unit tests of advantage math.
- `experiments/manifests/<run>/` per-run directory pattern, `config_snapshot.json`, `train_log.json`, `eval_log.json`.
- Test layout under `tests/unit` and `tests/smoke`.

**Rewrite:**

- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/baseline/policy.py` — replace `SoftmaxPolicy` with `LoRAPolicy` wrapping Qwen2.5-1.5B + PEFT. Keep `sample_text_action` / `greedy_text_action` interface so the trainer/evaluator contracts survive.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/hgpo/{objective.py,policy.py,grouping.py}` — replace bandit-shaping logic with real H-GRPO advantage math (`Â_traj`, `Â_turn`, `Â_H`, consistency loss).
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/trainers/train.py` — replace toy episode loop with K-trajectory rollout collector + GRPO update.
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/scripts/run_modal_train.sh` — replace with Modal Python apps.

**New modules to create** (paths under `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/`):

- `src/policy/lora_policy.py` — Qwen2.5-1.5B + PEFT LoRA, exposes `forward_logprobs(input_ids, action_mask)`, `merge_and_save()`, `load_adapter()`.
- `src/policy/vllm_runner.py` — local vLLM `LLM` instance; `generate(prompts, n=K, sampling_params)`; `sync_weights_from_lora(lora_state_dict)`.
- `src/algorithms/grpo/trainer.py` — main `HGPOTrainer` class with `train_step(batch_of_K_trajectories)`, KL controller, AdamW, gradient accumulation.
- `src/algorithms/grpo/rollout.py` — `collect_K_trajectories(env_pool, prompt, K, max_turns)`; returns a `TrajectoryGroup` dataclass (per-turn observations, actions, action token ids, action logprobs under rollout policy, per-turn raw rewards, final reward).
- `src/algorithms/grpo/advantage.py` — pure-functional `compute_traj_advantages(group)`, `compute_turn_advantages(group, decomposer)`, `combine(alpha, traj_adv, turn_adv)`, `consistency_loss(lambda, traj_adv, turn_adv)`. **Unit-tested in isolation.**
- `src/algorithms/grpo/loss.py` — clipped PPO surrogate over per-token logprobs masked by action mask, weighted by per-turn advantage broadcast to tokens; KL term.
- `src/algorithms/hgpo/decomposers/base.py` — `class TurnRewardDecomposer(Protocol): def decompose(group: TrajectoryGroup) -> per_turn_rewards`.
- `src/algorithms/hgpo/decomposers/progress.py` — Method C: WebShop product-match-score delta; ALFWorld subgoal delta.
- `src/algorithms/hgpo/decomposers/judge.py` — Method A: HTTP client to judge service; in-process LRU + persistent sqlite cache; prompt templates per env.
- `src/algorithms/hgpo/decomposers/turnrd.py` — Method B: model + inference path. Training script lives in `src/turnrd/train.py`.
- `src/turnrd/model.py` — small Transformer (4 layers, hidden 256, [CLS] head); MSE-on-R (Mode 1) + MSE-on-judge (Mode 2).
- `src/turnrd/dataset.py` — builds (per-turn-embeddings, R) / (per-turn-embeddings, judge_scores) records from a trajectory replay buffer (jsonl on Volume).
- `src/turnrd/train.py` — `train_turnrd(replay_path, mode, ...)` standalone entrypoint; called from main trainer every N episodes.
- `src/judge/server.py` — Modal-deployable FastAPI/HTTP wrapper around vLLM(Qwen2.5-7B-Instruct), `/score_turns` endpoint, batches concurrent requests.
- `src/judge/cache.py` — sqlite-backed cache keyed by `(task_id, turn_idx, prefix_hash, judge_model_tag)`.
- `src/judge/prompts.py` — env-specific judge prompt templates (WebShop, ALFWorld) + score-parsing + normalization (`Σ r̂_t = R`).
- `modal/image.py` — Modal `Image.debian_slim()` definition: torch, transformers, peft, trl, vllm, accelerate, datasets, pydantic, fastapi, sqlite3, plus `pip install -e webshop`, plus ALFWorld data download to `/data/alfworld`.
- `modal/app_train.py` — Modal app: `@app.function(gpu="A100-80GB", volumes={"/vol": vol})` running `src.trainers.train_hgpo`.
- `modal/app_judge.py` — Modal app: `@app.cls(gpu="A10G", concurrency_limit=8, volumes={"/vol": vol})` serving judge.
- `modal/app_eval.py` — Modal app for eval-only on a checkpoint.
- `src/trainers/train_hgpo.py` — new top-level entrypoint replacing `src/trainers/train.py`.
- `configs/qwen15_lora.yaml`, `configs/grpo_train.yaml`, `configs/hgpo_progress.yaml`, `configs/hgpo_judge.yaml`, `configs/hgpo_turnrd.yaml`, `configs/judge_qwen7b.yaml`, `configs/env_webshop_llm.yaml`, `configs/env_alfworld_llm.yaml`.
- Tests:
  - `tests/unit/test_hgpo_advantage.py` — handcrafted K=4 trajectory groups with known answers for `Â_traj`, `Â_turn`, combine, consistency loss; α=1 reduces to GRPO.
  - `tests/unit/test_judge_cache.py` — hit/miss/persist.
  - `tests/unit/test_turnrd_model.py` — forward shapes, loss decreasing on synthetic data.
  - `tests/unit/test_progress_decomposer.py` — WebShop-style score-delta arithmetic.
  - `tests/integration/test_grpo_step_smoke.py` — one trainer step on a tiny Qwen0.5B + dummy env (gated on GPU presence).

---

## Phase plan

### Week 1 — 5/4 → 5/10: foundation + flat GRPO baseline working on WebShop

- Day 1–2: Build Modal image (`modal/image.py`); install WebShop + bake the 1.18M-product index into a Modal Volume (`/vol/webshop-data`); install ALFWorld; verify `WebShopAdapter.reset()` and `ALFWorldAdapter.reset()` succeed inside the image.
- Day 3: `src/policy/lora_policy.py` + `src/policy/vllm_runner.py` with a weight-sync helper. End-to-end check: 4 prompts → 4 generations from policy vLLM after a fake LoRA update.
- Day 4: `src/algorithms/grpo/rollout.py` — multi-turn collector for WebShop. Verify K=4 trajectories collected, with action token ids + logprobs preserved.
- Day 5: `src/algorithms/grpo/advantage.py` — trajectory-level only (turn-level stub returns zeros). Unit test coverage.
- Day 6: `src/algorithms/grpo/loss.py` + `trainer.py` — PPO clipped objective, KL to ref, AdamW, grad accumulation. End-to-end 50-episode flat-GRPO run on WebShop in Modal.
- Day 7: Eval harness via `src/trainers/evaluator.py` refactor + checkpoint reload via `src/trainers/io_utils.py`; baseline run committed to `experiments/manifests/baseline_grpo_webshop_<ts>/`.

### Week 2 — 5/11 → 5/17: H-GRPO Method C (free) + Method A (judge)

- Day 8: Add `compute_turn_advantages` + `combine` + `consistency_loss` in `advantage.py`; wire α and λ into trainer config. Verify α=1 gives bit-identical loss to flat GRPO on a fixed seed.
- Day 9: `src/algorithms/hgpo/decomposers/progress.py` (Method C). Run H-GRPO-Progress on WebShop, K=4, 100 episodes. Initial result.
- Day 10: `src/judge/server.py` + `cache.py` + `prompts.py`. Deploy `modal/app_judge.py` (Qwen2.5-7B-Instruct on A10G). Manual smoke: ≥10 trajectories scored, cache hits on rerun.
- Day 11: `src/algorithms/hgpo/decomposers/judge.py` integration. Async batching from trainer. End-to-end H-GRPO-Judge run on WebShop.
- Day 12: Begin TurnRD: `src/turnrd/model.py` + `dataset.py`. Unit tests pass.
- Day 13: `src/turnrd/train.py` (Mode 1 = predict R; Mode 2 = distill judge scores). Standalone training run on a 200-episode replay buffer collected from earlier runs.
- Day 14: `src/algorithms/hgpo/decomposers/turnrd.py` integration into trainer; periodic refresh hook every N=20 episodes.

### Week 3 — 5/18 → 5/22: Initial results + milestone deliverable

- Day 15–16: Three full WebShop runs (baseline / Method A / Method B), 1 seed each, 200 episodes. Save train+eval logs.
- Day 17: Plotting utility `src/reports/plotting.py` (matplotlib): reward curve, success rate, gradient-variance proxy. Also: regression check that α=1 reproduces baseline trajectory.
- Day 18–19: Write `reports/milestone/milestone.tex` (1 page) using the template from the guidelines: experiments done, hypothesis-status, week-4-5 plan, AI Tools Disclosure. PDF compile + sanity-check fits 1 page.
- Day 20: Buffer for debugging.
- Day 21: **Submit milestone PDF to Gradescope by 5/22 9pm.**

### Week 4 — 5/23 → 5/29: ALFWorld + ablations

- Day 22–23: ALFWorld prompt template + finalize adapter; baseline + best H-GRPO method on ALFWorld; Method C uses subgoal-completion delta.
- Day 24–25: Ablations: α ∈ {0, 0.5, 1}, consistency reg on/off, Method A/B/C side-by-side, TurnRD Mode 1 vs Mode 2, refresh cadence (frozen vs every 20 vs every 50 episodes).
- Day 26: ArCHer-style learned-critic baseline (small value head over hidden states, GAE) — even a partial implementation is enough for one comparison row.
- Day 27: Sample-efficiency analysis (steps to 30% success), gradient-variance analysis (per-token gradient-norm variance averaged across an epoch).

### Week 5 — 5/30 → 6/3: Poster

- Day 28: ALFWorld top-4 methods × 1 seed run completes.
- Day 29–30: Build poster (`reports/poster/poster.tex`, 24×36 landscape, 7 sections per guidelines): problem / motivation / prior work / method / findings / takeaways / what's left. AI Tools Disclosure.
- Day 31: **Submit poster + video by 6/3 9am, present at session.**

### Week 6 — 6/4 → 6/8: Final report

- Day 32–34: Write `reports/final/report.tex` — 1-page extended abstract + 8-page main paper. Include all ablations, gradient-variance plots, failure-mode analysis (per guidelines: "we expect analysis of the failure modes").
- Day 35: Updated team contributions section (per guidelines Section 7).
- Day 36: **Submit final report by 6/8 9pm.**

---

## Critical files

**Files to be modified:**

- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/baseline/policy.py` (replace SoftmaxPolicy with LoRAPolicy)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/hgpo/objective.py` (replace bandit shaping with real H-GRPO advantage math — moved into `src/algorithms/grpo/advantage.py`)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/hgpo/policy.py` (delete; replaced by LoRAPolicy + trainer)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/algorithms/hgpo/grouping.py` (delete; concept doesn't apply)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/trainers/train.py` (replaced by `train_hgpo.py`)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/trainers/evaluator.py` (extend EvalResult fields, accept LoRAPolicy)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/webshop_adapter.py` (add `format_prompt(state)` + `progress_score(info)`)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/alfworld_adapter.py` (add `format_prompt(state)` + `subgoal_count(info)`)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/factory.py` (no functional change, exposed-prompt helper)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/requirements.txt` (add torch, transformers, peft, trl, vllm, accelerate, datasets, modal, fastapi, sqlite3-via-stdlib, matplotlib)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/scripts/run_modal_train.sh` and `run_modal_eval.sh` (delete; replaced by Modal apps + `meta`-style README)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/README.md` (replace "next implementation steps" with current architecture + how-to-run)
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/configs/*.json|yaml` (add LLM/LoRA/HGPO configs; keep toy configs for unit-test runs)

**Files to be created:**

- `src/policy/{__init__.py, lora_policy.py, vllm_runner.py}`
- `src/algorithms/grpo/{__init__.py, trainer.py, rollout.py, advantage.py, loss.py, kl.py}`
- `src/algorithms/hgpo/decomposers/{__init__.py, base.py, progress.py, judge.py, turnrd.py}`
- `src/turnrd/{__init__.py, model.py, dataset.py, train.py}`
- `src/judge/{__init__.py, server.py, cache.py, prompts.py}`
- `src/reports/{__init__.py, plotting.py, tables.py}`
- `src/trainers/train_hgpo.py`
- `modal/{__init__.py, image.py, app_train.py, app_judge.py, app_eval.py, common.py}`
- `tests/unit/{test_hgpo_advantage.py, test_judge_cache.py, test_turnrd_model.py, test_progress_decomposer.py}`
- `tests/integration/test_grpo_step_smoke.py`
- `reports/milestone/milestone.tex`, `reports/poster/poster.tex`, `reports/final/report.tex`

---

## Verification

### Continuous (every PR)
- `pytest tests/unit -q` must pass; coverage must include the advantage math, the cache, the TurnRD model forward, and the progress decomposer.
- Type-check with `pyright` (or skip if not preferred).

### Algorithmic correctness
1. **GRPO recovery**: With α=1 and λ=0, `HGPOTrainer.compute_loss` must produce numerically identical loss/grad to a separately-coded vanilla flat-GRPO trainer on a fixed-seed batch of 4 toy trajectories. Asserted in `tests/unit/test_hgpo_advantage.py`.
2. **Group normalization**: Mean of `Â_traj` across K=4 trajectories ≈ 0; std ≈ 1 (modulo σ-floor).
3. **Consistency loss sign**: Synthetic `Â_turn` summing to `Â_traj` should give zero consistency loss; perturbed should be positive.
4. **Cache correctness**: A second judge call for the same `(task_id, turn_idx, prefix_hash)` returns the cached score without re-querying the model (assertable via a mock judge backend).

### End-to-end
1. **Toy multi-turn run**: 50 episodes on a toy 4-action multi-turn env using a tiny Qwen0.5B; baseline reward must improve over random and beat α=0.5 H-GRPO only when reward signal is informative (sanity).
2. **WebShop baseline run**: 200 episodes flat GRPO improves over no-train ReAct baseline by ≥3% absolute success rate.
3. **WebShop H-GRPO-Progress run**: At α=0.5, expected to match or beat flat GRPO; if not, document as falsifying evidence (acceptable per proposal Section 3.3 "Falsification").
4. **Reproducibility**: Two runs with same seed produce identical loss/eval-return curves (verify against the reproducibility controls already in `src/trainers/io_utils.py`).
5. **Modal cost gate**: Track `$ spent` per run via Modal dashboard; baseline 200-ep WebShop run must stay under ~$15 (≈4 GPU-hr on A100-80G).

### Deliverable gates
- 5/22: `reports/milestone/milestone.pdf` exists, ≤ 1 page, includes ≥ 1 figure, includes AI Tools Disclosure.
- 6/3: `reports/poster/poster.pdf` exists, 24×36 landscape, covers all 7 required questions from guidelines Section 6.
- 6/8: `reports/final/report.pdf` exists, ≤ ~9 pages (1 abstract + ~8 main), includes updated team contributions + AI Tools Disclosure.

---

## Risk register (and mitigations)

| Risk | Mitigation |
|---|---|
| WebShop install on Modal flaky | Bake into image at build time; pin commit; volume-snapshot the data once. |
| ALFWorld data download is 5GB+ | Cache to Volume on first run; gate ALFWorld behind milestone. |
| vLLM weight-sync mismatch with HF training graph | Start with HF generate fallback; add vLLM after baseline runs end-to-end. (Even though we picked vLLM from day 1, Day 3 includes a fallback path.) |
| Judge latency dominates training time | Aggressive cache + `concurrency_limit` on Modal app; option to fall back to Method C while debugging. |
| TurnRD overfits to small replay buffer | Mode 2 (distill judge) starts from much more signal-dense labels; refresh cadence keeps it on-policy. |
| 1× A100-80G OOMs with batch×K rollouts | Gradient accumulation; truncate rollout context; reduce K to 2 for ALFWorld if needed. |
| Schedule slip past 5/22 | Milestone fallback: submit with **only** baseline + Method C (Progress) — both fully working with no judge dependency. |
