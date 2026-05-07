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

## Structure

```
configs/                        Method/env JSON configs (one per method × env)
src/algorithms/grpo/            Trainer, advantage math, KL controller, collectors
src/algorithms/hgpo/decomposers/{progress,judge,turnrd,counterfactual}
src/policy/                     LoRAPolicy, VLLMRunner, weight sync
src/turnrd/                     TurnRD model + dataset + embedder + standalone trainer
src/judge/                      Judge backends + cache
src/envs/                       WebShop + ALFWorld adapters
infra/                          Modal apps (train_loop, train_turnrd, eval, image, common)
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

# 4. (Only if running LLMJudge) provision the OpenAI secret on Modal
modal secret create openai-secret OPENAI_API_KEY=sk-...
```

Full walkthrough with troubleshooting: `docs/MODAL_SETUP.md`.

**Shared resources** (already provisioned on the team's Modal workspace):

| Resource | Path / name |
|---|---|
| Modal Volume | `cs224r-hgpo-vol` (mounted at `/vol/` inside every Modal function) |
| WebShop SFT adapter | `/vol/checkpoints/sft_v3_20260504_154752` |
| ALFWorld SFT adapter | `/vol/checkpoints/sft_alfworld_v1_<ts>` (ask Joseph for current ts) |
| Replay/ckpt cache | `/vol/cache/<MethodName>/{replay.jsonl,ckpt.pt}` |
| Per-run logs | `/vol/manifests/<run_name>_<ts>/train_log.json` |

---

## 2. Run training + eval

Every method's eval is **baked into the training run**: each round
finishes with a held-out greedy-K=1 pass on `[eval_task_id_base,
eval_task_id_base + eval_episodes)`. Default `eval_task_id_base=6500`,
`eval_episodes=50`. Disjoint from any seed's training slice by
construction → apples-to-apples across methods + seeds.

### 2a. Canonical sweep (one shell, one method or all)

`scripts/run_methods_protocol.sh` is the **canonical launcher**.
Defaults: `--seed 11 --rounds 5 --eps-per-round 40`,
`--sft-adapter /vol/checkpoints/sft_v3_20260504_154752`. WebShop only.

```bash
# Full 6-method WebShop bake-off, seed 11
bash scripts/run_methods_protocol.sh --seed 11

# Subset (e.g. teammate B owns LLMJudge + Progressive)
bash scripts/run_methods_protocol.sh --seed 11 \
  --methods LLMJudge,Progressive

# Different seed for a second teammate (slices are disjoint)
bash scripts/run_methods_protocol.sh --seed 23 \
  --methods flatGRPO,TurnRDV1,TurnRDV2

# Print commands without executing
bash scripts/run_methods_protocol.sh --seed 11 --dry-run
```

Flags exposed:

| Flag | Default | Notes |
|---|---|---|
| `--seed` | `11` | Drives `task_id_offset = seed * rounds * eps_per_round` |
| `--rounds` | `5` | numOfRound — number of train_loop ↔ train_turnrd alternations (or just train_loop rounds for non-TurnRD methods) |
| `--eps-per-round` | `40` | H-GRPO episodes per round |
| `--sft-adapter` | `/vol/checkpoints/sft_v3_20260504_154752` | Warm-start LoRA adapter on the Volume |
| `--methods` | all 6 | CSV: `SFTOnly,flatGRPO,TurnRDV1,TurnRDV2,Progressive,LLMJudge` |
| `--dry-run` | off | Prints the underlying `modal run` commands |

Each method blocks until its rounds finish. Run under `nohup` if you want the launcher to outlive your shell:

```bash
nohup bash scripts/run_methods_protocol.sh --seed 11 \
  > /tmp/methods_seed11.log 2>&1 &
```

### 2b. Per-method commands (bypass the launcher)

Use these if you want to run a single method with non-default flags
(e.g. fewer rounds for a quick check). These are exactly the
invocations the launcher dispatches.

**Common values used below** (override per teammate):
```
SEED=11
ROUNDS=5
EPS_PER_ROUND=40
SFT_ADAPTER=/vol/checkpoints/sft_v3_20260504_154752
EVAL_EPS=50
EVAL_BASE=6500
BASE_OFFSET=$(( SEED * ROUNDS * EPS_PER_ROUND ))
```

#### TurnRDV1 / TurnRDV2 — orchestrator (`scripts/run_turnrd_modal.py`)

The orchestrator interleaves the parent H-GRPO loop (writes a replay
buffer + reads the latest TurnRD ckpt) with the standalone TurnRD
trainer (reads the buffer + writes the ckpt) round by round.

```bash
# TurnRDV1
scripts/run_turnrd_modal.py \
  --config configs/TurnRDV1.json \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/TurnRDV1/replay.jsonl \
  --ckpt-path   /vol/cache/TurnRDV1/ckpt.pt \
  --run-name-prefix TurnRDV1 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --eval-episodes 50 --eval-task-id-base 6500

# TurnRDV2 — adds --carry-policy-across-rounds
scripts/run_turnrd_modal.py \
  --config configs/TurnRDV2.json \
  --seed 11 --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --replay-path /vol/cache/TurnRDV2/replay.jsonl \
  --ckpt-path   /vol/cache/TurnRDV2/ckpt.pt \
  --run-name-prefix TurnRDV2 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
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
modal run infra/app_train_loop.py::train_loop_webshop \
  --config /workspace/configs/SFTOnly.json \
  --n-episodes 0 --k 4 --max-turns 6 \
  --task-id-offset $(( 11 * 5 * 40 )) \
  --run-name SFTOnly_seed11 --round-idx 0 \
  --sft-adapter /vol/checkpoints/sft_v3_20260504_154752 \
  --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.30
```

`total_episodes=0` (in `SFTOnly.json`) makes the train loop skip the
RL body and go straight to the held-out eval pass.

#### flatGRPO / Progressive / LLMJudge — per-round Modal loop

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
the other two methods. (LLMJudge requires the `openai-secret` Modal
Secret; see setup step 4.)

### 2c. ALFWorld

ALFWorld config support is partial — only three methods have a ready
`_alfworld.json`:

| Method | ALFWorld config |
|---|---|
| Progressive | `configs/method_hgpo_progress_alfworld.json` |
| TurnRDV2 | `configs/method_hgpo_turnrd_v2_alfworld.json` |
| TurnRD lean (≈ V1) | `configs/method_hgpo_turnrd_lean_alfworld.json` |

For **SFTOnly / flatGRPO / LLMJudge / TurnRDV1**, you'll need to
create analogous `*_alfworld.json` configs (clone the WebShop config
and adjust `env_name`, `max_turns=30`, `gpu_mem_util=0.20`); flag
Joseph if you need this.

For the 3 supported ALFWorld methods, use the existing parallel
launcher:

```bash
# Launches all 3 methods under nohup, in parallel.
# Per-method logs: /tmp/alfworld_sft_sweep_{method_b_v2,method_b_lean,method_c}.log
bash scripts/run_alfworld_sweep_with_sft.sh /vol/checkpoints/sft_alfworld_v1_<ts>
```

Or invoke a single method by hand:

```bash
# TurnRDV2 on ALFWorld
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

# Progressive on ALFWorld (per-round modal-run loop, max_turns=30)
SEED=11; ROUNDS=5; EPS=40
for r in $(seq 0 $((ROUNDS-1))); do
  OFFSET=$(( SEED * ROUNDS * EPS + r * EPS ))
  modal run infra/app_train_loop.py::train_loop_alfworld \
    --config /workspace/configs/method_hgpo_progress_alfworld.json \
    --n-episodes ${EPS} --k 4 --max-turns 30 \
    --task-id-offset ${OFFSET} \
    --run-name Progressive_alfworld_seed${SEED}_round$(printf '%02d' $r) \
    --round-idx ${r} \
    --sft-adapter /vol/checkpoints/sft_alfworld_v1_<ts> \
    --eval-episodes 50 --eval-task-id-base 6500 --gpu-mem-util 0.20
done
```

### 2d. Cost / wall-clock estimates

| Method | Cost / round | Wall / round | 5×40 total |
|---|---|---|---|
| SFTOnly | ~$1.50 (eval only) | ~5 min | ~$1.50 |
| flatGRPO | ~$5 | ~13 min | ~$25 |
| Progressive | ~$5 | ~13 min | ~$25 |
| LLMJudge | ~$6 + judge $$ | ~15 min | ~$30 + judge |
| TurnRDV1 | ~$8 (loop+fit) | ~20 min | ~$40 |
| TurnRDV2 | ~$8 (loop+fit) | ~20 min | ~$40 |

Full bake-off (all 6 methods, single seed): **~$160, ~3 hr wall**. Two
teammates running disjoint method subsets in parallel halves the wall
time. See `docs/METHOD_B_SWEEP_INTEGRATION.md` for the underlying
estimates.

---

## 3. Pull artifacts locally

Each round writes `train_log.json` to
`/vol/manifests/<run_name>_<ts>/`. Pull the per-round dirs into the
local repo:

```bash
# TurnRDV2 example (4 rounds, seed 11) — adjust prefix + count for other methods
mkdir -p experiments/manifests/_TurnRDV2_seed11
modal volume ls cs224r-hgpo-vol /manifests | grep TurnRDV2_seed11
# pick the timestamps printed above, then for each:
for ts_dir in TurnRDV2_seed11_round00_<ts0> TurnRDV2_seed11_round01_<ts1> ... ; do
  mkdir -p experiments/manifests/_TurnRDV2_seed11/$ts_dir
  modal volume get cs224r-hgpo-vol /manifests/$ts_dir/train_log.json \
    experiments/manifests/_TurnRDV2_seed11/$ts_dir/train_log.json --force
done
```

For non-TurnRD methods the run name pattern is `<METHOD>_seed<S>_round<NN>_<ts>`.

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

### Per-round eval table (CSV-style for the report)

`experiments/manifests/methods_comparison.json` already records the
canonical per-round eval for SFTOnly / flatGRPO / Progressive /
LLMJudge / TurnRDV1 / TurnRDV2 at seed=11. New seeds should append
entries with the same schema (`n_rounds`, `best_eval_return`,
`mean_pct_success`, `_per_round_eval[]`).

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

| Owner | WebShop methods | ALFWorld methods | Seeds |
|---|---|---|---|
| Joseph | TurnRDV1, TurnRDV2 | TurnRDV2 (existing config) | 11, 23 |
| Teammate B | flatGRPO, Progressive | Progressive (existing config) | 11, 23 |
| Teammate C | SFTOnly, LLMJudge | (create configs first) | 11, 23 |

Each teammate runs e.g.:

```bash
# Teammate B, WebShop seed 11
nohup bash scripts/run_methods_protocol.sh --seed 11 \
  --methods flatGRPO,Progressive \
  > /tmp/teammate_b_seed11.log 2>&1 &
```

Then `modal volume get` your method's per-round dirs into
`experiments/manifests/_<Method>_seed<S>/`, run
`scripts/plot_protocol_comparison.py` once everyone's logs are local,
and add a `<Method>` entry to
`experiments/manifests/methods_comparison.json` with the per-round
eval block.
