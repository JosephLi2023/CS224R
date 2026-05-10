# CS224R H-GRPO

Hierarchical Group Relative Policy Optimization (H-GRPO) on text-action
RL environments (WebShop, ALFWorld), with a bake-off across **6 methods**:

| Method | Decomposer | What's different |
|---|---|---|
| **SFTOnly** | n/a (eval-only) | No RL update — just the held-out eval pass against the SFT warm-start adapter. Reference floor. |
| **flatGRPO** | progress (no-op) | H-GRPO with `alpha=1.0` so the per-turn signal is dropped. Trajectory-level GRPO baseline. |
| **LLMJudge** | judge | OpenAI gpt-4o-mini judge produces per-turn rewards, cached in SQLite. |
| **TurnRDV1** | turnrd v1 | Original learned [CLS] cross-attention decomposer, causal mask, lean variant. |
| **TurnRDV2** | turnrd v2 | Bidirectional + Σ α·v identifiable + progress-prior init, with `--carry-policy-across-rounds`. |
| **Progressive** | progress | Parameter-free progress decomposer (env raw_env_reward delta). |

Trainer stack: PEFT LoRA on `Qwen/Qwen2.5-1.5B-Instruct`, vLLM for
rollout, K-grouped PPO-clipped surrogate + Schulman k3 KL +
AdaptiveKLController. All methods share `HGPOTrainer`; only the
decomposer changes (or the algorithm wrapper in flatGRPO's case).

---

## Headline results — AlfWorld Tier-2 SOTA (`pct_success`, n=100 held-out eval)

The Tier-2 sweep (TurnRDV2 with `lr=5e-6`, `K=8`, skip-dead-K guard, α
ablation) **improves the prior best by +12 pp absolute** (0.46 → 0.58)
and beats the SFT-warm-start anchor by **+18 pp**. All 3 α-variants
escape the prior 46% noise-floor plateau by ≥10 pp.

| Rank | Method | best | last | mean | n_rounds | total eval eps |
|---|---|---:|---:|---:|---:|---:|
| 1 | **TurnRDV2_a050** (Tier-2, α=0.50) | **0.580** | 0.580 | **0.540** | 5 | 500 |
| 2 | TurnRDV2_a075 (Tier-2, α=0.75) | 0.580 | 0.580 | 0.518 | 5 | 500 |
| 3 | TurnRDV2_a025 (Tier-2, α=0.25) | 0.560 | 0.560 | 0.514 | 5 | 500 |
| 4 | TurnRDV1 (prior K=4/lr=1e-6) | 0.460 | 0.460 | 0.435 | 4 | 200 |
| 5 | flatGRPO (prior K=4/lr=1e-6) | 0.460 | 0.420 | 0.436 | 5 | 250 |
| 6 | TurnRDV2 (prior, same α=0.50 at K=4/lr=1e-6) | 0.440 | 0.440 | 0.430 | 4 | 200 |
| 7 | Progressive (prior) | 0.420 | 0.400 | 0.410 | 2 | 100 |
| 8 | SFTOnly (RL-free anchor) | 0.400 | 0.400 | 0.400 | 1 | 50 |

**Recommended config:** `α=0.50` — top-ranked on both `best` (tied at
0.58) and the more decision-relevant `mean` aggregate (0.540). The α
distribution is unimodal (mean: 0.514 → 0.540 → 0.518), arguing against
further sweeping.

**The Tier-2 fix package is the dominant lever** — same α=0.50 blend at
K=4/lr=1e-6 only achieved 0.44; the lr/K/skip-dead-K combo at K=8/lr=5e-6
takes it to 0.58 (+14 pp from the same blend coefficient alone).

### What's in the Tier-2 fix package

| Knob | Tier-2 sweep | Prior K=4/lr=1e-6 sweep |
|---|---|---|
| α (H-GRPO blend) | {0.25, 0.50, 0.75} | 0.5 / 1.0 |
| Learning rate | 5e-6 (5×) | 1e-6 |
| K (trajectories per task) | 8 (2×) | 4 |
| Skip-dead-K guard | enabled | not present |
| Eval episodes | 100 (n=100, σ ≈ 5pp) | 50 (n=50, σ ≈ 7pp) |

The skip-dead-K guard short-circuits the heavy 3-forward-pass policy
update on K-groups where all final rewards are equal (zero PG signal by
construction). This was firing on 44-57% of K-groups in the prior
sweep — the guard recovers ~12 min of wallclock per round at K=8 while
preserving the rollout's per-turn rewards for TurnRD's standalone
trainer.

### Reproduction

```bash
# Wave 1 (smoke test, ~50 min): launches α=0.5 only at K=8 to verify no OOM
bash scripts/run_alfworld_alpha_sweep.sh \
    /vol/checkpoints/sft_alfworld_v1_20260507_165617 --smoke-only

# Wave 2 (~50 min wallclock for both in parallel): launch α=0.25 + α=0.75
bash scripts/run_alfworld_alpha_sweep.sh \
    /vol/checkpoints/sft_alfworld_v1_20260507_165617 --rest

# Monitor live
python3 scripts/monitor_alfworld_alpha_sweep.py --watch 60

# Aggregate to manifest + pull locally
modal run infra/app_aggregate_alfworld.py
modal volume get cs224r-hgpo-vol /manifests/4method_comparison_alfworld.json \
    experiments/manifests/4method_comparison_alfworld.json --force

# Regenerate plots + analysis report
.venv/bin/python scripts/analyze_alfworld_alpha_sweep.py
```

Full ablation analysis (mechanistic explanation of why α=0.5 wins, plus
the 4-panel headline figure and Tier-4 diagnostics): see
[`reports/alfworld_alpha_sweep_README.md`](reports/alfworld_alpha_sweep_README.md).

Aggregated manifest: [`experiments/manifests/4method_comparison_alfworld.json`](experiments/manifests/4method_comparison_alfworld.json) (8 keys).

---

## Structure

```
configs/                        Method/env JSON configs (one per method × env)
src/algorithms/grpo/            Trainer, advantage math, KL controller, collectors
src/algorithms/hgpo/decomposers/{progress,judge,turnrd,counterfactual}
src/policy/                     LoRAPolicy, VLLMRunner, weight sync
src/turnrd/                     TurnRD model + dataset + embedder + standalone trainer
src/judge/                      Judge backends + cache
src/envs/                       WebShop + ALFWorld adapters
src/datasets/                   sft_webshop / sft_alfworld loaders
infra/                          Modal apps (install, sft_train, train_loop, train_turnrd, eval)
scripts/                        Orchestrators + aggregators + plotters
tests/                          Unit + smoke + integration tests
docs/                           User-facing guides
experiments/manifests/          Per-run train_log.json, summary.json, methods_comparison.json
```

---

## 1. One-time setup (every teammate)

```bash
# 1. Clone + venv
cd /path/to/CS224R
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install modal

# 2. Modal auth (browser-based; one-time per machine)
modal token new
modal token current        # sanity check

# 3. CPU smoke — first run will build the Modal image (~10 min, cached forever)
modal run infra/app_train.py::hello
```

Full walkthrough with troubleshooting: `docs/MODAL_SETUP.md`.

### 1.4 OpenAI API key (only required for `LLMJudge`)

The `LLMJudge` method calls **OpenAI gpt-4o-mini** as the per-turn
reward model. The key reaches the Modal container as a Modal Secret
named `openai-secret` (env var `OPENAI_API_KEY`). The other 5 methods
do **not** need a key — you can skip this section unless your task
list includes `LLMJudge`.

**Cost expectation**: a full 5×40 LLMJudge run makes ~4800 judge
calls. With gpt-4o-mini at $0.15 / $0.60 per 1M input/output tokens
and ~500 input + 50 output tokens per call, total **~$0.50 per run**
on top of the ~$30 Modal A100 cost. Calls are cached in
`/vol/cache/judge.sqlite` so repeat episodes are free.

**Step 1 — Check whether the secret is already provisioned** (it's
shared across the workspace; you may not need to add anything):

```bash
modal secret list
# Expected if already set:
#   ┃ Name          ┃ Created at           ┃ Created by ┃ Last used at  ┃
#   │ openai-secret │ 2026-05-04 10:34 PDT │ <teammate> │ <recent>      │
```

If `openai-secret` is present, **skip to Step 4 (verify)**. Otherwise
continue.

**Step 2 — Get an OpenAI API key**

1. Go to <https://platform.openai.com/api-keys> (sign in / create an
   account if you don't have one).
2. Click **"Create new secret key"**, scope it to `gpt-4o-mini` (or
   leave it unrestricted for personal accounts).
3. **Copy the `sk-...` value immediately** — OpenAI shows it only once.
4. Make sure your account has a payment method on file
   (<https://platform.openai.com/account/billing>) — gpt-4o-mini is
   pay-as-you-go and will fail at first call without billing
   configured. ~$5 of credit is more than enough for the bake-off.

**Step 3 — Provision the secret on Modal**

```bash
# The secret name MUST be `openai-secret` and the payload key MUST
# be `OPENAI_API_KEY` — both are hardcoded in
# `infra/common.py::OPENAI_SECRET_NAME` and
# `infra/common.py::OPENAI_SECRET_REQUIRED_KEYS`.
modal secret create openai-secret OPENAI_API_KEY=sk-proj-xxxxxxxx...

# Verify it landed
modal secret list | grep openai-secret
```

**Step 4 — Verify the key reaches a Modal container**

The provided probe function reads `OPENAI_API_KEY` from the container
env and reports whether it's set:

```bash
modal run infra/app_train.py::env_probe
# Expected to print (among other keys):
#   'openai_api_key_present': True
```

If `False`, the most common causes are:
- The secret is named something other than `openai-secret` (Modal is
  case-sensitive).
- The payload key is `OPENAI_KEY` instead of `OPENAI_API_KEY`. Recreate
  the secret: `modal secret create openai-secret OPENAI_API_KEY=sk-...`.
- You set `CS224R_SKIP_OPENAI_SECRET=1` in your shell — `unset` it.

**Step 5 — (Optional) Opt out if you're only running non-LLMJudge methods**

If you're not running `LLMJudge` and don't want to provision the
secret, opt out of the secret attachment so deploys / runs don't fail
at function-decorator time:

```bash
export CS224R_SKIP_OPENAI_SECRET=1
modal run infra/app_train_loop.py::train_loop_webshop ...
```

This skips the `secrets=maybe_openai_secret()` mount on every
function. `LLMJudge` will then fail at first OpenAI call with a clear
missing-key error — the other 5 methods are unaffected.

For more detail (live cost dashboard, Modal-side flags, etc.) see
`docs/MODAL_SETUP.md` § 7.

**Shared resources** (provisioned once on the team's Modal workspace,
then reused by every run):

| Resource | Path / name | Provisioning step |
|---|---|---|
| Modal Volume | `cs224r-hgpo-vol` (mounted at `/vol/` inside every Modal function) | Auto-created on first `modal run` |
| WebShop env install | `/vol/code/webshop`, `/vol/data/webshop/{indexes_1k,resources_1k}` | Section 1.5a (~$2, ~30 min, one-time) |
| WebShop SFT trajectories | `/vol/data/webshop/human_trajs/` (~50 trajectories) | Section 1.5a |
| WebShop SFT adapter | `/vol/checkpoints/sft_v3_<ts>/` (current: `sft_v3_20260504_154752`) | Section 1.5b (~$0.50, ~10 min) |
| ALFWorld data | `/vol/data/alfworld/json_2.1.1/{train,valid_seen,valid_unseen}/` | Section 1.5c (~$1, ~5 min) |
| ALFWorld SFT trajectories | `/vol/data/alfworld/sft_trajs.jsonl` | Section 1.5c (~$1, ~15-30 min) |
| ALFWorld SFT adapter | `/vol/checkpoints/sft_alfworld_v1_<ts>/` | Section 1.5d (~$0.50, ~30 min) |

The Volume is shared across all teammates' workspaces — once any
teammate provisions a resource it's visible to everyone. **Check first,
then provision only what's missing:**

```bash
# Check what's already on the Volume
modal volume ls cs224r-hgpo-vol /
modal volume ls cs224r-hgpo-vol /checkpoints | grep sft        # SFT adapters
modal volume ls cs224r-hgpo-vol /data/webshop                  # WebShop env data
modal volume ls cs224r-hgpo-vol /data/alfworld/json_2.1.1      # ALFWorld games
```

If `sft_v3_20260504_154752` (WebShop) and `sft_alfworld_v1_*` (ALFWorld)
both exist, you can **skip Section 1.5 entirely** and jump to Section 2.

### 1.5 Provisioning (only if a resource above is missing)

#### 1.5a. WebShop env + human-trajectory SFT data

```bash
# Install WebShop into the Volume (~$2, ~30 min)
modal run infra/app_webshop_install.py --action pip_install
modal run infra/app_webshop_install.py --action download_spacy
modal run infra/app_webshop_install.py --action build_index_1k

# Pull the ~50 human trajectories used for SFT (~$0, ~1 min)
modal run infra/app_data.py --action download_human_trajs

# Sanity check
modal run infra/app_data.py --action summarize_sft_succeeded
# Expected: ~745 (prompt, action) examples with reward ≥ 0.5
```

#### 1.5b. WebShop SFT adapter

```bash
# Trains LoRA on the human trajectories (~$0.50, ~10 min on A100).
# Outputs: /vol/checkpoints/sft_v1_<ts>/
modal run --detach infra/app_sft_train.py --epochs 3 --min-reward 0.5

# Discover the timestamp of your fresh adapter:
modal volume ls cs224r-hgpo-vol /checkpoints | grep sft_v1
# Set this in every subsequent run via --sft-adapter:
SFT_WEBSHOP=/vol/checkpoints/sft_v1_<ts>
```

To match the existing `methods_comparison.json` baselines exactly, use
the team's published adapter:

```bash
SFT_WEBSHOP=/vol/checkpoints/sft_v3_20260504_154752
```

#### 1.5c. ALFWorld data + SFT trajectories

```bash
# Download AlfWorld TextWorld games + THOR scene assets (~$1, ~5 min)
modal run infra/app_alfworld_install.py --action download

# Verify (should list >100 game directories)
modal volume ls cs224r-hgpo-vol /data/alfworld/json_2.1.1/train

# Generate SFT trajectories using the PDDL expert (~$1, ~15-30 min CPU)
# Writes /vol/data/alfworld/sft_trajs.jsonl
modal run --detach infra/app_alfworld_sft_gen.py
modal run infra/app_alfworld_sft_gen.py --action inspect    # peek + cardinality
```

#### 1.5d. ALFWorld SFT adapter

```bash
# (~$0.50, ~30 min on A100). Outputs /vol/checkpoints/sft_alfworld_v1_<ts>/
modal run --detach infra/app_sft_train_alfworld.py --epochs 3 --min-reward 0.5

# Discover + set:
modal volume ls cs224r-hgpo-vol /checkpoints | grep sft_alfworld
SFT_ALFWORLD=/vol/checkpoints/sft_alfworld_v1_<ts>
```

---

## 2. Run training + eval

Every method's eval is **baked into the training run**: each round
finishes with a held-out greedy-K=1 pass on `[eval_task_id_base,
eval_task_id_base + eval_episodes)`. Default `eval_task_id_base=6500`,
`eval_episodes=50`. Disjoint from any seed's training slice by
construction → apples-to-apples across methods + seeds.

### 2a. WebShop — canonical sweep (one shell, one method or all)

`scripts/run_methods_protocol.sh` is the **canonical WebShop launcher**.
Defaults: `--seed 11 --rounds 5 --eps-per-round 40`,
`--sft-adapter /vol/checkpoints/sft_v3_20260504_154752`.

```bash
# Full 6-method WebShop bake-off, seed 11
bash scripts/run_methods_protocol.sh --seed 11

# Subset (e.g. teammate B owns LLMJudge + Progressive)
bash scripts/run_methods_protocol.sh --seed 11 \
  --methods LLMJudge,Progressive

# Different seed for a second teammate (slices are disjoint)
bash scripts/run_methods_protocol.sh --seed 23 \
  --methods flatGRPO,TurnRDV1,TurnRDV2

# Custom SFT adapter (if you trained your own in 1.5b)
bash scripts/run_methods_protocol.sh --seed 11 \
  --sft-adapter /vol/checkpoints/sft_v1_<ts>

# Print commands without executing
bash scripts/run_methods_protocol.sh --seed 11 --dry-run
```

Flags exposed:

| Flag | Default | Notes |
|---|---|---|
| `--seed` | `11` | Drives `task_id_offset = seed * rounds * eps_per_round` |
| `--rounds` | `5` | numOfRound |
| `--eps-per-round` | `40` | H-GRPO episodes per round |
| `--sft-adapter` | `/vol/checkpoints/sft_v3_20260504_154752` | Warm-start LoRA adapter on the Volume |
| `--methods` | all 6 | CSV: `SFTOnly,flatGRPO,TurnRDV1,TurnRDV2,Progressive,LLMJudge` |
| `--dry-run` | off | Prints the underlying `modal run` commands |

Each method blocks until its rounds finish. Run under `nohup` if you want the launcher to outlive your shell:

```bash
nohup bash scripts/run_methods_protocol.sh --seed 11 \
  > /tmp/methods_seed11.log 2>&1 &
```

### 2b. ALFWorld — canonical sweep

`scripts/run_alfworld_sweep_with_sft.sh` is the canonical ALFWorld
launcher. **Launches 5 methods in parallel**, each in its own nohup
process so the local terminal can be closed. Methods supported:
TurnRDV2, TurnRDV1, Progressive, flatGRPO, SFTOnly.
*(LLMJudge ALFWorld config is not yet committed — flag Joseph if needed.)*

```bash
# Full 5-method ALFWorld parallel sweep
bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>

# Override defaults via env vars
N_ROUNDS=4 EPS_PER_ROUND=40 SEED=11 EVAL_EPS=50 \
  bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>

# Tail all 5 logs at once
tail -f /tmp/alfworld_sft_sweep_{TurnRDV2,TurnRDV1,Progressive,flatGRPO,SFTOnly}.log
```

Wall-clock budget: ~3-6 hr; ~$33 total ($30 RL + $3 SFTOnly).
Per-method logs: `/tmp/alfworld_sft_sweep_<MethodName>.log`.

Env vars exposed:

| Var | Default | Notes |
|---|---|---|
| `N_ROUNDS` | `5` | numOfRound |
| `EPS_PER_ROUND` | `40` | |
| `TURNRD_EPOCHS` | `3` | Standalone TurnRD epochs between rounds |
| `SEED` | `11` | |
| `EVAL_EPS` | `50` | |
| `EVAL_TASK_BASE` | `6500` | |

### 2c. Per-method commands (bypass the launchers)

Use these if you want to run a single method with non-default flags
(e.g. fewer rounds for a quick check). These are exactly the
invocations the launchers dispatch.

**Common values used below** (override per teammate):
```
SEED=11
ROUNDS=5
EPS_PER_ROUND=40
SFT_WEBSHOP=/vol/checkpoints/sft_v3_20260504_154752
SFT_ALFWORLD=/vol/checkpoints/sft_alfworld_v1_<ts>
EVAL_EPS=50
EVAL_BASE=6500
BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
```

#### TurnRDV1 / TurnRDV2 — orchestrator (`scripts/run_turnrd_modal.py`)

The orchestrator interleaves the parent H-GRPO loop (writes a replay
buffer + reads the latest TurnRD ckpt) with the standalone TurnRD
trainer (reads the buffer + writes the ckpt) round by round.

**WebShop:**
```bash
# TurnRDV1 — WebShop
scripts/run_turnrd_modal.py \
  --config configs/TurnRDV1.json \
  --env-name webshop \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/TurnRDV1/replay.jsonl \
  --ckpt-path   /vol/cache/TurnRDV1/ckpt.pt \
  --run-name-prefix TurnRDV1 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --eval-episodes 50 --eval-task-id-base 6500

# TurnRDV2 — WebShop (adds --carry-policy-across-rounds)
scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2.json \
  --env-name webshop \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/TurnRDV2/replay.jsonl \
  --ckpt-path   /vol/cache/TurnRDV2/ckpt.pt \
  --run-name-prefix TurnRDV2 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --eval-episodes 50 --eval-task-id-base 6500 \
  --carry-policy-across-rounds
```

**ALFWorld:**
```bash
# TurnRDV1 — ALFWorld
scripts/run_turnrd_modal.py \
  --config configs/method_hgpo_turnrd_lean_alfworld.json \
  --env-name alfworld \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/method_b_lean_alfworld/replay.jsonl \
  --ckpt-path   /vol/cache/method_b_lean_alfworld/ckpt.pt \
  --run-name-prefix TurnRDV1_alfworld \
  --sft-adapter /vol/checkpoints/sft_alfworld_v1_<ts> \
  --eval-episodes 50 --eval-task-id-base 6500 \
  --carry-policy-across-rounds

# TurnRDV2 — ALFWorld
scripts/run_turnrd_modal.py \
  --config configs/method_hgpo_turnrd_v2_alfworld.json \
  --env-name alfworld \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/method_b_v2_alfworld/replay.jsonl \
  --ckpt-path   /vol/cache/method_b_v2_alfworld/ckpt.pt \
  --run-name-prefix TurnRDV2_alfworld \
  --sft-adapter /vol/checkpoints/sft_alfworld_v1_<ts> \
  --eval-episodes 50 --eval-task-id-base 6500 \
  --carry-policy-across-rounds
```

Selected `run_turnrd_modal.py` flags (full list: `scripts/run_turnrd_modal.py --help`):

| Flag | Default | Notes |
|---|---|---|
| `--rounds` | `5` | numOfRound |
| `--episodes-per-round` | `40` | |
| `--turnrd-epochs` | `3` | Standalone TurnRD epochs between rounds |
| `--env-name` | `webshop` | `webshop` or `alfworld` (routes to `train_loop_<env>`) |
| `--seed` | none | Drives `task_id_offset = seed * rounds * episodes_per_round` |
| `--carry-policy-across-rounds` | off | **Required for TurnRDV2.** Round 0 loads SFT; round N≥1 loads previous round's saved adapter. |
| `--adapter-dir` | `/vol/checkpoints` | Where per-round adapters land |
| `--eval-episodes` | `50` | 0 to skip the held-out pass |
| `--eval-task-id-base` | `6500` | Disjoint from training slice; ≤6910 for WebShop |
| `--dry-run` | off | Print the per-round commands only |

#### SFTOnly — single Modal call (no RL)

```bash
# WebShop
modal run infra/app_train_loop.py::train_loop_webshop \
  --config /workspace/configs/SFTOnly.json \
  --n-episodes 0 --k 4 --max-turns 6 \
  --task-id-offset $(( 11 * 5 * 40 )) \
  --run-name SFTOnly_seed11 --round-idx 0 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.30

# ALFWorld (max_turns=30, gpu_mem_util=0.30; uses train_loop_alfworld)
modal run infra/app_train_loop.py::train_loop_alfworld \
  --config /workspace/configs/SFTOnly_alfworld.json \
  --n-episodes 0 --k 4 --max-turns 30 \
  --task-id-offset 0 \
  --run-name SFTOnly_alfworld_seed11_round00 --round-idx 0 \
  --sft-adapter /vol/checkpoints/sft_alfworld_v1_<ts> \
  --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.30
```

`total_episodes=0` (in `SFTOnly.json`) makes the train loop skip the
RL body and go straight to the held-out eval pass.

#### flatGRPO / Progressive / LLMJudge — per-round Modal loop

**WebShop:**
```bash
SEED=11; ROUNDS=5; EPS=40
for r in $(seq 0 $((ROUNDS-1))); do
  OFFSET=$(( SEED * ROUNDS * EPS + r * EPS ))
  modal run infra/app_train_loop.py::train_loop_webshop \
    --config /workspace/configs/Progressive.json \
    --n-episodes ${EPS} --k 4 --max-turns 6 \
    --task-id-offset ${OFFSET} \
    --run-name Progressive_seed${SEED}_round$(printf '%02d' $r) \
    --round-idx ${r} \
    --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
    --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.30
done
```

Replace `Progressive.json` with `flatGRPO.json` or `LLMJudge.json` for
the other two. (LLMJudge requires the `openai-secret` Modal Secret
provisioned in setup step 4.)

**ALFWorld** (`max_turns=30`, `gpu_mem_util=0.20`, `train_loop_alfworld`,
threads `--save-adapter-out` so each round inherits the previous round's
adapter — matches `run_alfworld_sweep_with_sft.sh`):

```bash
SEED=11; ROUNDS=5; EPS=40; SFT_ALFWORLD=/vol/checkpoints/sft_alfworld_v1_<ts>
RUN_PREFIX=Progressive_alfworld_seed${SEED}            # or flatGRPO_alfworld_seed${SEED}
CONFIG=configs/method_hgpo_progress_alfworld.json      # or configs/flatGRPO_alfworld.json
INLINE_BASE_OFFSET=$(( SEED * ROUNDS * EPS ))
ADAPTER_DIR=/vol/checkpoints

for r in $(seq 0 $((ROUNDS-1))); do
  OFFSET=$(( INLINE_BASE_OFFSET + r * EPS ))
  RUN_NAME=${RUN_PREFIX}_round$(printf '%02d' $r)
  if [[ $r -eq 0 ]]; then
    LOAD_ADAPTER=${SFT_ALFWORLD}
  else
    LOAD_ADAPTER=${ADAPTER_DIR}/${RUN_PREFIX}_round$(printf '%02d' $((r-1)))_adapter
  fi
  modal run --detach infra/app_train_loop.py::train_loop_alfworld \
    --config /workspace/${CONFIG} \
    --n-episodes ${EPS} --k 4 --max-turns 30 \
    --task-id-offset ${OFFSET} \
    --run-name ${RUN_NAME} --round-idx ${r} \
    --sft-adapter ${LOAD_ADAPTER} \
    --save-adapter-out ${ADAPTER_DIR}/${RUN_NAME}_adapter \
    --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.20
done
```

### 2d. ALFWorld config availability

| Method | WebShop config | ALFWorld config |
|---|---|---|
| SFTOnly | `configs/SFTOnly.json` | `configs/SFTOnly_alfworld.json` |
| flatGRPO | `configs/flatGRPO.json` | `configs/flatGRPO_alfworld.json` |
| Progressive | `configs/Progressive.json` | `configs/method_hgpo_progress_alfworld.json` |
| TurnRDV1 | `configs/TurnRDV1.json` | `configs/method_hgpo_turnrd_lean_alfworld.json` |
| TurnRDV2 | `configs/TurnRDV2.json` | `configs/method_hgpo_turnrd_v2_alfworld.json` |
| **LLMJudge** | `configs/LLMJudge.json` | **Not yet — clone `LLMJudge.json` and set `env`-related fields, or skip LLMJudge for ALFWorld** |

### 2e. Cost / wall-clock estimates

| Method | Cost / round | Wall / round | 5×40 total |
|---|---|---|---|
| SFTOnly | ~$1.50 (eval only) | ~5 min | ~$1.50 |
| flatGRPO | ~$5 | ~13 min | ~$25 |
| Progressive | ~$5 | ~13 min | ~$25 |
| LLMJudge | ~$6 + judge $$ | ~15 min | ~$30 + judge |
| TurnRDV1 | ~$8 (loop+fit) | ~20 min | ~$40 |
| TurnRDV2 | ~$8 (loop+fit) | ~20 min | ~$40 |

Full WebShop bake-off (all 6 methods, single seed): **~$160, ~3 hr
wall**. Full ALFWorld 5-method parallel sweep: **~$33, ~3-6 hr wall**.
Two teammates running disjoint method subsets in parallel halves the
wall time. See `docs/METHOD_B_SWEEP_INTEGRATION.md` for underlying
estimates.

---

## 3. Pull artifacts locally

Each round writes `train_log.json` to
`/vol/manifests/<run_name>_<ts>/`. Pull the per-round dirs into the
local repo:

```bash
# TurnRDV2 example (5 rounds, seed 11) — adjust prefix + count for other methods
mkdir -p experiments/manifests/_TurnRDV2_seed11
modal volume ls cs224r-hgpo-vol /manifests | grep TurnRDV2_seed11
# pick the timestamps printed above, then for each:
for ts_dir in TurnRDV2_seed11_round00_<ts0> TurnRDV2_seed11_round01_<ts1> ... ; do
  mkdir -p experiments/manifests/_TurnRDV2_seed11/$ts_dir
  modal volume get cs224r-hgpo-vol /manifests/$ts_dir/train_log.json \
    experiments/manifests/_TurnRDV2_seed11/$ts_dir/train_log.json --force
done
```

Run-name pattern: `<METHOD>_seed<S>_round<NN>_<ts>` for WebShop;
`<METHOD>_alfworld_seed<S>_round<NN>_<ts>` for ALFWorld.

---

## 4. Aggregate + plot for the milestone report

### Merge per-round logs (TurnRD-style methods)

`scripts/merge_turnrd_round_logs.py` concatenates a method's per-round
`train_log.json` files into a single contiguous reward curve with the
plotter-compatible shape:

```bash
.venv/bin/python scripts/merge_turnrd_round_logs.py \
  --manifests-dir experiments/manifests/_TurnRDV2_seed11 \
  --seed 11 \
  --run-name-prefix TurnRDV2 \
  --out experiments/manifests/_TurnRDV2_seed11/TurnRDV2_seed11_merged.json
```

### Single-run reward curve

```bash
.venv/bin/python scripts/plot_reward_curve.py \
  experiments/manifests/_TurnRDV2_seed11/TurnRDV2_seed11_merged.json \
  --out reports/TurnRDV2_seed11_curve.png
```
Top panel: per-episode mean R ± 1σ + MA(5). Bottom panel: KL coef +
grad_norm + observed_kl.

### Side-by-side method comparison (for the report)

`scripts/plot_protocol_comparison.py` accepts `--method label=path`
where `path` is a single `train_log.json` OR a directory of round
dirs (auto-merged on the fly).

```bash
.venv/bin/python scripts/plot_protocol_comparison.py \
  --method 'SFTOnly=experiments/manifests/_SFTOnly_seed11/.../train_log.json' \
  --method 'flatGRPO=experiments/manifests/_flatGRPO_seed11/' \
  --method 'LLMJudge=experiments/manifests/_LLMJudge_seed11/' \
  --method 'TurnRDV1=experiments/manifests/_TurnRDV1_seed11/' \
  --method 'TurnRDV2=experiments/manifests/_TurnRDV2_seed11/' \
  --method 'Progressive=experiments/manifests/_Progressive_seed11/' \
  --turnrd-diagnostics \
  --out reports/methods_comparison_seed11.png
```

3-panel figure:
- **Top**: per-episode training reward MA(5), one line per method
- **Middle**: held-out eval `avg_return` markers — one dot per round per method
- **Bottom** (if `--turnrd-diagnostics`): `cls_query_norm` + `alpha_var` trajectories for any TurnRD method

### Per-round eval table for the report

`experiments/manifests/methods_comparison.json` records the canonical
per-round eval for the WebShop bake-off. New seeds / new env runs
should append entries with the same schema (`n_rounds`,
`best_eval_return`, `mean_pct_success`, `_per_round_eval[]`).

---

## 5. Pre-flight smoke (before you spend $)

Cheap end-to-end checks before launching a full sweep:

```bash
# (a) CPU image + Volume sanity (~$0)
modal run infra/app_train.py::hello

# (b) A100 + library probe (~$0.05)
modal run infra/app_train.py::env_probe

# (c) TurnRD producer↔trainer end-to-end on real Qwen (1×2 ep, ~$1, ~5 min)
nohup .venv/bin/python scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2.json \
  --rounds 1 --episodes-per-round 2 --turnrd-epochs 1 \
  --seed 11 --run-name-prefix _smoke \
  --carry-policy-across-rounds \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --replay-path /vol/cache/TurnRDV2/replay.jsonl \
  --ckpt-path   /vol/cache/TurnRDV2/ckpt.pt \
  --eval-episodes 0 \
  --adapter-dir /vol/checkpoints/_smoke \
  > /tmp/smoke.log 2>&1 &

# (d) Print a non-TurnRD method's per-round commands without running
bash scripts/run_methods_protocol.sh --methods Progressive --dry-run
```

---

## 6. Tests

```bash
.venv/bin/python -m pytest tests/unit/                  # fast, local-only
.venv/bin/python -m pytest tests/smoke/                 # local smoke (no Modal)
.venv/bin/python -m pytest tests/integration/           # may require Modal/secret
```

---

## 7. Reproducibility controls

- **Per-seed task disjointness** — `task_id_offset = seed * rounds * episodes_per_round`; eval always on `[6500, 6550)` regardless of seed.
- **Per-run config snapshot** — every Modal call writes the exact flags + JSON config it received into the run dir's `summary.json`.
- **Volume-backed cache** — vLLM HF cache, judge SQLite, replay JSONL, ckpts all live on `cs224r-hgpo-vol`.
- **Detached Modal jobs** — orchestrators use `--detach` so cloud jobs survive local CLI death; they poll `modal app list` for cross-round sequencing.

---

## 8. Documentation

| File | Purpose |
|---|---|
| `docs/MODAL_SETUP.md` | Modal account → token → first smoke. Walkthrough with troubleshooting. |
| `docs/METHOD_B_SWEEP_INTEGRATION.md` | TurnRD orchestrator design, cost estimates, sanity-check sequence, failure-mode reference. |
| `docs/HGPO_TRAINING_LOOP.md` | The H-GRPO trainer math + decomposer interface. |
| `docs/method_naming.md` | Old (`method_b_*`) ↔ new (`TurnRDV2`, etc.) name map for legacy artifacts. |

---

## 9. Who runs what (suggested split for the milestone)

Assumes the team-shared SFT adapters
(`/vol/checkpoints/sft_v3_20260504_154752` for WebShop and
`/vol/checkpoints/sft_alfworld_v1_<ts>` for ALFWorld) are already on
the Volume. If not, the owner of each env runs Section 1.5 first.

| Owner | Env | Methods | Seeds | One-line invocation |
|---|---|---|---|---|
| Joseph | WebShop | TurnRDV1, TurnRDV2 | 11, 23 | `nohup bash scripts/run_methods_protocol.sh --seed 11 --methods TurnRDV1,TurnRDV2 > /tmp/joseph_ws_11.log 2>&1 &` |
| Teammate B | WebShop | flatGRPO, Progressive | 11, 23 | `nohup bash scripts/run_methods_protocol.sh --seed 11 --methods flatGRPO,Progressive > /tmp/B_ws_11.log 2>&1 &` |
| Teammate C | WebShop | SFTOnly, LLMJudge | 11, 23 | `nohup bash scripts/run_methods_protocol.sh --seed 11 --methods SFTOnly,LLMJudge > /tmp/C_ws_11.log 2>&1 &` |
| Joseph | ALFWorld | TurnRDV2, TurnRDV1 | 11 | `bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>` (this script runs all 5 methods in parallel; pick PIDs from output if you want to kill TurnRDV2/TurnRDV1 only) |
| Teammate B | ALFWorld | Progressive, flatGRPO | 11 | (same script as above; logs at `/tmp/alfworld_sft_sweep_{Progressive,flatGRPO}.log`) |
| Teammate C | ALFWorld | SFTOnly | 11 | (same script as above; log at `/tmp/alfworld_sft_sweep_SFTOnly.log`) |

After your method finishes:

1. **Pull logs locally** (Section 3) into `experiments/manifests/_<Method>_seed<S>/`
2. **Aggregate** if it's a TurnRD method (`scripts/merge_turnrd_round_logs.py`)
3. **Run the comparison plot** once everyone's logs are local (Section 4 final command) — this needs all 6 methods' artifacts to overlay
4. **Append a `<Method>` entry** to `experiments/manifests/methods_comparison.json` with the per-round eval block
