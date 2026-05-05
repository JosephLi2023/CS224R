# CS224R H-GRPO

Hierarchical Group Relative Policy Optimization (H-GRPO) on text-action
RL environments (WebShop, ALFWorld), with three turn-decomposer
variants:

- **Method A (Judge)** — OpenAI/vLLM judge produces per-turn rewards
  (cached SQLite-backed).
- **Method B (TurnRD)** — learned [CLS] cross-attention decomposer
  fitted on a JSONL replay buffer; refresh-loaded into the H-GRPO
  trainer per `cfg.turnrd.refresh_every_episodes`.
- **Method C (Progress)** — uses the env's raw per-turn reward as
  decomposition.

Trainer stack: PEFT LoRA on `Qwen/Qwen2.5-1.5B-Instruct`, vLLM for
rollout, K-grouped PPO-clipped surrogate + Schulman k3 KL +
AdaptiveKLController. All three methods share the same
`HGPOTrainer`; the only difference is the decomposer plugged in.

---

## Structure

```
configs/                        Method/env/eval JSON configs
src/
  algorithms/grpo/              Trainer, advantage math, KL controller, collectors
  algorithms/hgpo/decomposers/  {progress, judge, turnrd} per-turn decomposers
  policy/                       LoRAPolicy, VLLMRunner, weight sync
  turnrd/                       TurnRD model + dataset + embedder + standalone trainer
  judge/                        Judge backends + cache
  envs/                         WebShop + ALFWorld adapters
  trainers/                     train_hgpo (config loader), evaluator
infra/                          Modal apps (train_loop, train_turnrd, eval, image, common)
scripts/                        Orchestrators + aggregators + plotters
tests/                          Unit + smoke tests
docs/                           User-facing guides
experiments/manifests/          Per-run train_log.json, eval_log.json, summary.json
```

---

## Setup

```bash
cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # local-only deps
pip install modal                        # Modal CLI for cloud runs
modal token new                          # browser-based auth (one-time)
```

Modal-side image deps (`torch==2.4.0`, `transformers==4.45.2`,
`vllm==0.6.3.post1`, `peft==0.13.2`, …) are baked into the Modal image
via `infra/image.py`; they're not installed locally.

See `docs/MODAL_SETUP.md` for Modal authentication, Volume, and Secret
setup.

---

## Run Training

### Method B (TurnRD) — full protocol

The orchestrator coordinates the parent H-GRPO loop (writes a replay
buffer + reads the latest TurnRD ckpt) with the standalone TurnRD
trainer (reads the buffer + writes the ckpt) round-by-round:

```bash
scripts/run_turnrd_modal.py \
  --rounds 5 --episodes-per-round 40 --turnrd-epochs 3 \
  --seed 11 \
  --sft-adapter /vol/checkpoints/<sft_run_dir>
```

Defaults: `--config configs/method_hgpo_turnrd.json`, replay path
`/vol/cache/turnrd_replay.jsonl`, ckpt path `/vol/cache/turnrd_ckpt.pt`,
held-out eval `--eval-episodes 50 --eval-task-id-base 10000`.

For a cheap end-to-end sanity check before the full run:

```bash
modal run infra/app_train_step_turnrd.py            # ~$0.20, 3-4 min
scripts/run_turnrd_modal.py --dry-run               # free; prints commands
```

See `docs/METHOD_B_SWEEP_INTEGRATION.md` for the design rationale +
cost estimates.

### Method A (Judge) / Method C (Progress)

```bash
modal run infra/app_train_loop.py \
  --config configs/method_hgpo_judge.json \
  --n-episodes 200 --k 4 --task-id-offset 0 \
  --run-name method_a_seed11 \
  --sft-adapter /vol/checkpoints/<sft_run_dir> \
  --eval-episodes 50 --eval-task-id-base 10000

modal run infra/app_train_loop.py \
  --config configs/method_hgpo_progress.json \
  --n-episodes 200 --k 4 --task-id-offset 0 \
  --run-name method_c_seed11 \
  --sft-adapter /vol/checkpoints/<sft_run_dir> \
  --eval-episodes 50 --eval-task-id-base 10000
```

Each run drops a per-episode `train_log.json` + per-run `summary.json`
under `/vol/manifests/<run_name>_<ts>/`.

### Run all six methods (sweep)

```bash
bash scripts/run_webshop_protocol.sh --seed 11   # 6 methods × 1 seed
bash scripts/run_webshop_protocol.sh --seed 23
```

---

## Run Evaluation

`infra/app_train_loop.py::train_loop_smoke` accepts `--eval-episodes N
--eval-task-id-base B` and appends an `eval` block to `train_log.json`
after the K rounds of training. Greedy K=1 sampling on a held-out task
range; same env as training. Disjoint, fixed range across rounds +
methods + seeds → apples-to-apples by construction. Default 50 eps on
`[10000, 10050)`.

---

## Aggregate + Plot

Method B writes per-round artifacts under
`experiments/manifests/method_b_orchestrated_seed{N}_round{NN}_<ts>/`.

```bash
# Concatenate per-round train_logs into a single contiguous reward curve.
scripts/merge_turnrd_round_logs.py \
  --seed 11 \
  --out experiments/manifests/method_b_seed11_merged/train_log.json

# Side-by-side comparison vs flat-GRPO baseline.
scripts/plot_protocol_comparison.py \
  --method "flat_grpo=/vol/manifests/grpo_<ts>/train_log.json" \
  --method "method_b=experiments/manifests" \
  --turnrd-diagnostics \
  --out comparison_seed11.png

# Single-run reward curve for any train_log.json.
scripts/plot_reward_curve.py path/to/train_log.json --out curve.png
```

The 3-panel comparison plot shows: (top) per-episode training reward
moving average, (middle) held-out eval markers per round per method,
(bottom) TurnRD-specific diagnostics (cls_query L2 norm + alpha
variance) when present.

---

## Tests

```bash
.venv/bin/python -m pytest tests/unit/ tests/smoke/
```

---

## Reproducibility Controls

- **Per-seed task disjointness** — `seed * rounds * episodes_per_round`
  base offset; eval always on the protocol-reserved range
  `[10000, 10050)` regardless of seed.
- **Per-run config snapshot** — every Modal call writes the exact
  flag + JSON config it received into the run dir's `summary.json`.
- **Volume-backed cache** — vLLM HF cache, judge SQLite, replay JSONL,
  ckpt all live on the shared `cs224r-hgpo-vol` Modal Volume.
- **Detached Modal jobs** — `--detach` on every `modal run` so cloud
  jobs survive local CLI death; orchestrator polls `modal app list` for
  cross-round sequencing.

---

## Documentation

| File | Purpose |
|---|---|
| `docs/METHOD_B_SWEEP_INTEGRATION.md` | How Method B integrates with the protocol sweep (cost estimates, sanity-check sequence, failure-mode reference). |
| `docs/MODAL_SETUP.md` | Modal authentication, Volume, Secret setup. |
