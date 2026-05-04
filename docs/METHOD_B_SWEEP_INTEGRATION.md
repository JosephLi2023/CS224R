# Integrating Method B (TurnRD) into the WebShop Protocol Sweep

This guide explains how to add Method B to `scripts/run_webshop_protocol.sh`
so the existing 6-method × 2-seed protocol sweep dispatches it through the
new Modal coordination path (`scripts/run_turnrd_modal.py`) rather than
the local-Python fallback (`python -m src.trainers.train`) that all
methods currently share.

> **Why a special path for Method B?** Methods A (judge) and C (progress)
> have a single dispatcher: `infra/app_train_loop.py --config ...`. Method
> B has two coupled processes — the parent H-GRPO loop (writes a replay
> buffer + reads the latest TurnRD ckpt) and the standalone TurnRD
> trainer (reads the buffer + writes the ckpt). They alternate per round,
> and only the orchestration script (`scripts/run_turnrd_modal.py`)
> handles that coordination correctly. See the Method-B completion
> summary in `~/.llms/plans/cs224r_hgpo_execution.plan.md` for the
> design rationale.

---

## 1. Current state of `scripts/run_webshop_protocol.sh`

The launcher already knows about Method B — `method_hgpo_turnrd` is in
the `METHODS` array (line ~26) — but it dispatches every method
identically through the local-Python fallback:

```bash
PYTHONPATH=. python3 -m src.trainers.train \
  --env-config configs/env_webshop.json \
  --train-config "${CONFIG}" \
  --eval-config configs/eval.json
```

This works for Methods A and C in principle (single process), but for
Method B the local fallback never triggers the standalone TurnRD
trainer; it would just run the parent H-GRPO loop with a TurnRD
decomposer that trains from scratch on every refresh-cadence boundary
and never picks up an external ckpt. **Method B's whole point — letting
TurnRD fit on the cumulative replay buffer between H-GRPO rounds — is
lost.**

---

## 2. The 1-line integration

Patch the launcher to special-case `method_hgpo_turnrd`:

```bash
# In the `for METHOD in "${METHODS[@]}"` loop, replace the unconditional
# `PYTHONPATH=. python3 -m src.trainers.train ...` invocation with:

if [[ "${METHOD}" == "method_hgpo_turnrd" ]]; then
  # Method B: use the producer ↔ standalone-trainer orchestrator.
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: would invoke scripts/run_turnrd_modal.py --rounds 5 --episodes-per-round 40 --turnrd-epochs 3"
    continue
  fi
  scripts/run_turnrd_modal.py \
    --config "${CONFIG}" \
    --rounds 5 \
    --episodes-per-round 40 \
    --turnrd-epochs 3 \
    --run-name-prefix "${METHOD}_seed${SEED}"
else
  # Methods A/C and the baselines: existing dispatch path.
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY RUN: would invoke train with --train-config ${CONFIG} --env-config configs/env_webshop.json --seed ${SEED}"
    continue
  fi
  PYTHONPATH=. python3 -m src.trainers.train \
    --env-config configs/env_webshop.json \
    --train-config "${CONFIG}" \
    --eval-config configs/eval.json
fi
```

That's the only structural change. The pre-flight checks in
`scripts/run_turnrd_modal.py` will catch any config/path mismatches
before the first Modal call lands.

> **Note on the seed.** `scripts/run_turnrd_modal.py` does not yet
> accept a `--seed` flag — the seed lives inside the JSON config
> (`configs/method_hgpo_turnrd.json::run.seed`). For multi-seed runs,
> either: (a) add a `--seed` flag to the orchestrator that overrides
> the config's seed before launch, or (b) maintain per-seed config
> copies (`configs/method_hgpo_turnrd_seed11.json`,
> `configs/method_hgpo_turnrd_seed23.json`). Option (a) is the cleaner
> long-term path; option (b) is the smaller change today.

---

## 3. Multi-seed protocol layout

The existing protocol is **6 methods × 2 seeds = 12 runs**:

| Method | Config | Dispatcher |
|---|---|---|
| ReAct (no-train) | `method_react_eval.json` | `python -m src.trainers.train` |
| Flat GRPO | `method_flat_grpo.json` | `python -m src.trainers.train` |
| H-GRPO Method C (progress) | `method_hgpo_progress.json` | `python -m src.trainers.train` |
| H-GRPO Method A (judge) | `method_hgpo_judge.json` | `python -m src.trainers.train` |
| **H-GRPO Method B (TurnRD)** | **`method_hgpo_turnrd.json`** | **`scripts/run_turnrd_modal.py`** |
| ArCHer | `method_archer.json` | `python -m src.trainers.train` |

To run the full 12-run sweep:

```bash
# Two seeds, sequentially. Total wall-clock ~6-12 hours on Modal.
bash scripts/run_webshop_protocol.sh --seed 11
bash scripts/run_webshop_protocol.sh --seed 23
```

Each invocation drops 6 run directories under
`experiments/manifests/`. Method-B run directories are emitted by
each round of the orchestrator separately:
```
experiments/manifests/method_hgpo_turnrd_seed11_round00_<ts>/  (round 0 train_loop)
experiments/manifests/method_hgpo_turnrd_seed11_round01_<ts>/  (round 1 train_loop)
...
experiments/manifests/method_hgpo_turnrd_seed11_round04_<ts>/  (round 4 train_loop)
```
Plus the standalone-trainer run dirs (one per round) which get a
`turnrd_ckpt.pt` written into `/vol/cache/`. Aggregation tools that
expect a single `train_log.json` per method will need a small
post-processing step to concatenate the per-round logs into one
contiguous reward curve — see §6.

---

## 4. Cost + wall-clock estimates

| Method | A100-min per run | Cost per run (~$3/A100-hr) | Cost per seed (1 run) | Cost for 2-seed sweep |
|---|---|---|---|---|
| ReAct | ~10 (eval-only K=1) | ~$0.50 | ~$0.50 | ~$1.00 |
| Flat GRPO | ~30 | ~$1.50 | ~$1.50 | ~$3.00 |
| H-GRPO Progress (C) | ~30 | ~$1.50 | ~$1.50 | ~$3.00 |
| H-GRPO Judge (A, OpenAI) | ~35 + judge $5 | ~$7.00 | ~$7.00 | ~$14.00 |
| **H-GRPO TurnRD (B)** | **~45** (5 rounds × ~9 min) | **~$2.50** | **~$2.50** | **~$5.00** |
| ArCHer | ~35 | ~$1.75 | ~$1.75 | ~$3.50 |
| **Total** | | | **~$15** | **~$30** |

Method B's overhead vs Method C (~50% more wall-clock) comes from:
- **Standalone TurnRD trainer**: ~30s per round × 5 rounds = ~2.5 min total.
- **Cross-round container cold-start**: ~30s per round × 4 transitions = ~2 min.
- **Producer's per-rollout embedder forward**: ~5% on top of each
  H-GRPO step (negligible).

If wall-clock matters more than coordination granularity, raise
`--episodes-per-round` and lower `--rounds` correspondingly (e.g.
3 rounds × 67 eps each instead of 5 × 40). The trade-off is fewer
TurnRD refreshes, so the H-GRPO policy spends more episodes with a
stale decomposer.

---

## 5. Sanity-check sequence (REQUIRED before kicking off the sweep)

Run these three checks in order. Each one catches a different class of
failure cheaply.

### 5a. Local CPU smoke (free, ~5 sec)

Verifies the full Method-B loop works end-to-end without spending Modal
credits:

```bash
miniconda3/bin/python -m pytest tests/smoke/test_method_b_config_loader_smoke.py -v
```

Asserts: producer writes valid JSONL → dataset reads it → standalone
trainer writes a checkpoint → refresh fn loads it into the live
decomposer (`cls_query` byte-equality before vs after).

### 5b. Orchestrator dry-run (free, instant)

Catches the three silent-misconfig failure modes the orchestrator
guards against:

```bash
scripts/run_turnrd_modal.py --dry-run --rounds 1 --episodes-per-round 2 --turnrd-epochs 1
```

Should print 2 `modal run ...` commands without errors. If any of
these fail, fix the config first:
- `replay-path mismatch` → align `--replay-path` with `cfg.turnrd.replay_buffer_path`.
- `decomposer != turnrd` → using the wrong config; pass `--config configs/method_hgpo_turnrd.json`.
- `--rounds <= 0` → typo in flag.

### 5c. Modal A100 smoke (~$0.20, 3-4 min)

Verifies the C3 reattach + second-optimizer path works on real
hardware (Qwen2.5-1.5B + bf16 hidden states + LoRA + vLLM):

```bash
modal run infra/app_train_step_turnrd.py
```

The decisive assertion is `cls_query_delta > 0` — proves the C3
reattach is flowing gradient through TurnRD AND the second AdamW is
stepping. Other assertions cover: no-NaN total_loss, ≥ 1 LoRA
parameter moved, replay JSONL D matches policy hidden_size (1536).

If 5a + 5b + 5c all pass, the sweep is safe to launch.

---

## 6. Post-hoc analysis: aggregating per-round Method-B logs

Methods A/C produce one `train_log.json` per run dir (200 episodes in
one file). Method B produces 5 (one per round, 40 episodes each). To
merge them into a single reward curve for plotting:

```python
# scripts/merge_turnrd_round_logs.py (sketch — not yet written)
import json
import glob
from pathlib import Path

def merge(seed: int) -> dict:
    rows: list[dict] = []
    rounds = sorted(glob.glob(
        f"experiments/manifests/method_hgpo_turnrd_seed{seed}_round??_*"
    ))
    for round_dir in rounds:
        log = json.load(open(Path(round_dir) / "train_log.json"))
        # train_loop_smoke writes {"rows": [...], "config": {...}}
        for row in log["rows"]:
            # `row["episode"]` is per-round-local; offset by round size.
            row["global_episode"] = (
                int(round_dir.split("_round")[1].split("_")[0])
                * log["config"]["n_episodes"]
                + row["episode"]
            )
            rows.append(row)
    return {"rows": rows, "seed": seed}
```

The existing plotting script (`scripts/plot_reward_curve.py`) accepts a
single `train_log.json`-shaped input — feed it the merged dict above
and Method-B's reward curve plots alongside Methods A/C with no further
changes.

> Adding `merge_turnrd_round_logs.py` is a follow-up; it's a 30-line
> standalone script and unblocking it is purely an aggregation
> convenience, not a correctness issue.

---

## 7. Failure-mode quick reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Method B's reward curve is flat across rounds | TurnRD ckpt never reloaded | Verify `cfg.turnrd.ckpt_path` is set AND matches `--ckpt-path`; check `_refresh()` warning logs in container output |
| `total_loss` NaN partway through training | bf16 underflow in C3 reattach | Lower `cfg.hgpo.lambda_consistency` (default 0.1 → try 0.01); rerun with `cfg.train.kl_warmup_episodes > 0` |
| Standalone trainer OOM | Replay buffer too large | Cap with `--max-records N` on the standalone trainer, OR rotate the buffer between rounds (out-of-scope follow-up) |
| Producer JSONL grows unbounded | Default behavior — appends across rounds | Either accept it (default protocol stays under 500 MB) or add a per-round truncate to the orchestrator |
| Cross-round handoff appears decoupled | `cfg.turnrd.refresh_every_episodes == 0` | Set to a non-zero value (default 20 in `configs/method_hgpo_turnrd.json`) |
| Modal job fails with "missing tokenizer" | Embedder couldn't init | Confirm `LoRAPolicyConfig.cache_dir == /vol/hf_cache` AND volume is mounted |

---

## 8. References

- **Orchestrator script**: `scripts/run_turnrd_modal.py` (with `--help` for full CLI).
- **Modal apps**: `infra/app_train_loop.py` (parent H-GRPO + producer), `infra/app_train_turnrd.py` (standalone TurnRD fit), `infra/app_train_step_turnrd.py` (single-step smoke).
- **Config**: `configs/method_hgpo_turnrd.json`.
- **Loader**: `src/trainers/train_hgpo.py::build_trainer_from_config`.
- **CPU smoke**: `tests/smoke/test_method_b_config_loader_smoke.py`.
- **Modal setup**: `docs/MODAL_SETUP.md`.
- **Design rationale**: `~/.llms/plans/cs224r_hgpo_execution.plan.md` "Method-B completion summary" section, plus `MEDIUM_FIXES.md::M1` and `MEDIUM_FIXES.md::C3`.
