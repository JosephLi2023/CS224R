# WebShop SOTA pipeline (rank-32 + 7 MLP-target LoRA recipe)

End-to-end recipe for the AlfWorld → WebShop transplant: a deterministic
oracle SFT warm-start at LoRA rank-32 + 7 MLP target modules, then 3
RL methods (attention TurnRD, flat GRPO, LLM judge) bake-off against
the held-out eval slice `[6500, 6600)`.

All compute runs on **Modal** against a single shared volume
(`cs224r-hgpo-vol`). Local work is just shell + light Python.

---

## TL;DR

```bash
# One-time: install Modal CLI + auth
pip install modal && modal token new

# Phase 0: WebShop env install (idempotent; ~20 min the first time)
bash scripts/run_webshop_sft_v3_mlpr32.sh --skip-gen --train-only --dry-run  # sanity check paths
modal run infra/app_webshop_install.py --action pip_install
modal run infra/app_webshop_install.py --action download_spacy
modal run infra/app_webshop_install.py --action build_index_1k

# Optional (for `--include-human-trajs` concat): upstream gdown human trajs
modal run infra/app_data.py --action download_human_trajectories

# Phases 1-4: oracle gen → SFT train → SFT eval (single launcher, 4 steps)
bash scripts/run_webshop_sft_v3_mlpr32.sh --full
# → adapter lands at /vol/checkpoints/sft_webshop_v3_mlpr32_<TS>/

# Phase 5: wire the new adapter into the 3 RL launchers
sed -i '' "s/REPLACE_WITH_TS_FROM_PHASE4/<TS>/g" scripts/run_webshop_SOTA_*.sh

# Phase 6: launch the 4 RL methods in parallel
bash scripts/run_webshop_SOTA_attention_v1.sh
bash scripts/run_webshop_SOTA_flatGRPO_v1.sh
bash scripts/run_webshop_SOTA_LLMJudge_v1.sh
bash scripts/run_webshop_SOTA_Progress_v1.sh
```

Total wall-clock: ~10-15 hr Modal + ~30 min local. Total cost:
~$75-105 (SFT ~$25 + 4 RL methods × ~$20).

---

## Prerequisites

- **Modal account** with the `cs224r-hgpo-vol` volume + (optionally for
  LLMJudge) the `openai-secret` Secret containing `OPENAI_API_KEY`. To
  skip the secret (e.g. for attention + flatGRPO only) set
  `CS224R_SKIP_OPENAI_SECRET=1` before any `modal run`.
- **Local repo** with this README at the root. All `scripts/run_*.sh`
  invocations resolve paths relative to the repo root.

```bash
pip install modal           # Modal CLI
modal token new             # interactive browser auth
modal volume list           # confirm cs224r-hgpo-vol exists; create if not
modal volume create cs224r-hgpo-vol   # only if `list` doesn't show it
```

---

## Phase 0 — WebShop env install (one-time per volume)

Installs the upstream `web_agent_site` editable into the volume's
PEP-370 user-site, downloads the spaCy `en_core_web_lg` model, and
builds the BM25 Lucene index for the 1k-product dev split (the
default WebShop slice). Idempotent on re-runs.

```bash
modal run infra/app_webshop_install.py --action pip_install
modal run infra/app_webshop_install.py --action download_spacy
modal run infra/app_webshop_install.py --action build_index_1k
# Optional smoke: instantiate WebAgentTextEnv and call reset()
modal run infra/app_webshop_install.py --action reset_smoke
```

Wall-clock ~20-30 min the first time, near-instant on re-runs. Cost
~$1.

If you want the `--include-human-trajs` path in Phase 1 (concats the
upstream ~50 gdown human shopping trajectories into the SFT corpus),
also run:

```bash
modal run infra/app_data.py --action download_human_trajectories
```

---

## Phases 1-4 — SFT pipeline (oracle gen → train → eval)

Single launcher driving 4 Modal apps in sequence:

```bash
bash scripts/run_webshop_sft_v3_mlpr32.sh --full
```

What it does (defaults in parentheses):

1. **Install** — re-runs the Phase 0 install actions for idempotency.
2. **Gen** — `infra/app_webshop_sft_gen.py` runs a deterministic oracle
   over `N_SESSIONS=2000` distinct WebShop sessions on CPU. Per
   session: `search[<query>]` → walk up to 5 result pages → click
   target ASIN → best-effort option clicks for goal attributes →
   `click[Buy Now]`. Keeps only trajectories whose terminal env reward
   ≥ `REWARD_THRESHOLD=0.99`. With `INCLUDE_HUMAN_TRAJS=true` (default)
   it also concatenates the pre-rendered upstream gdown human
   trajectories. Output: `/vol/data/webshop/oracle_trajs.jsonl`
   (~10-15k rows). ~30-60 min, ~$0.50.
3. **Train** — `infra/app_sft_train.py` with `--lora-rank 32` +
   `--lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`,
   `--epochs 6`, `--learning-rate 5e-5`, `--max-seq-len 2048`,
   `--grad-accum 8`. Output:
   `/vol/checkpoints/sft_webshop_v3_mlpr32_<TS>/`. ~3-5 hr A100, ~$15-30.
4. **Eval** — `infra/app_train_loop.py::train_loop_webshop` with
   `configs/SFTOnly_webshop_mlpr32.json`, `--n-episodes 0`,
   `--eval-episodes 200`, `--eval-task-id-base 6500`. Confirms the
   adapter loads cleanly into a rank-32+MLP policy and prints
   pct_success on the held-out `[6500, 6700)` slice. ~30 min, ~$3-5.

### Modes

| Flag             | When to use |
|------------------|-------------|
| `--dry-run`      | Print every Modal command without launching (free).         |
| `--full`         | Run all 4 steps (default).                                  |
| `--skip-install` | Phase 0 already done.                                       |
| `--skip-gen`     | JSONL already exists at `DATA_PATH`.                        |
| `--train-only`   | Submit step 3 only (returns immediately; no eval).            |
| `--eval-only`    | Only step 4 (`ADAPTER_PATH=/vol/checkpoints/<run>_<ts>/latest`). |

### Tunable env-var knobs (defaults shown)

```bash
N_SESSIONS=2000  INCLUDE_HUMAN_TRAJS=true  REWARD_THRESHOLD=0.99
EPOCHS=6  LR=5e-5  MAX_SEQ_LEN=2048  GRAD_ACCUM=8  MIN_REWARD=0.5
EVAL_EPS=200  EVAL_TASK_BASE=6500
LORA_RANK=32  LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
RUN_TS=$(date +%Y%m%d_%H%M%S)
RUN_NAME=sft_webshop_v3_mlpr32_${RUN_TS}
DATA_PATH=/vol/data/webshop/oracle_trajs.jsonl
SAVE_EVERY_STEPS=500
RESUME_FROM=/vol/checkpoints/sft_webshop_v3_mlpr32_<ts>   # optional resume
```

### Resume after partial failure

| Crashed at         | Re-run with                                                |
|--------------------|------------------------------------------------------------|
| Install            | `--full` (install is idempotent).                          |
| Gen mid-way        | `--skip-install` (gen overwrites the JSONL fresh).         |
| Train (partial)    | `RESUME_FROM=/vol/checkpoints/<run>_<ts> bash ... --train-only` |
| Train (finished)   | `--eval-only ADAPTER_PATH=/vol/checkpoints/<run>_<ts>/latest` |
| Eval               | `--eval-only ADAPTER_PATH=/vol/checkpoints/<run>_<ts>/latest` |

### Sanity gates (block Phase 5-6 if any fail)

- **Gen** oracle win rate ≥ 50 % (else the attribute-availability of
  the 1k-product split is too sparse; revisit the oracle policy).
- **Train** loss decreases monotonically across epochs (no NaN/divergence).
- **Eval** `pct_success ≥ 0.42` on `[6500, 6700)`. The legacy rank-16
  baseline (`sft_v3_20260504_154752`) reportedly evals at ~0.45 on the
  same slice; we need within striking distance.

### Inspect intermediate outputs

```bash
# Peek at the oracle JSONL: cardinality, action mix, mean reward, first row.
modal run infra/app_webshop_sft_gen.py --action summarize \
    --output-path /vol/data/webshop/oracle_trajs.jsonl

# Pull the train log locally.
modal volume get cs224r-hgpo-vol \
    /checkpoints/sft_webshop_v3_mlpr32_<TS>/train_log.json train_log.json

# Pull the eval log locally.
modal volume get cs224r-hgpo-vol \
    /manifests/SFTOnly_webshop_mlpr32_eval_<TS>_*/train_log.json sft_eval.json
```

---

## Phase 5 — wire the new SFT adapter into the 3 RL launchers

The 3 RL launchers default `SFT_ADAPTER` to a placeholder
(`/vol/checkpoints/sft_webshop_v3_mlpr32_REPLACE_WITH_TS_FROM_PHASE4`).
After Phase 4 finishes, replace the placeholder with the actual
timestamp from the SFT train step. Two ways:

**A. One-shot sed (recommended):**

```bash
# Grab the TS from the SFT train step's Modal log; format YYYYMMDD_HHMMSS.
TS=20260524_181500   # ← replace with your actual timestamp
sed -i '' "s/REPLACE_WITH_TS_FROM_PHASE4/${TS}/g" scripts/run_webshop_SOTA_*.sh

# Verify all 3 launchers point at the new adapter:
grep "SFT_ADAPTER=" scripts/run_webshop_SOTA_*.sh
```

**B. Per-invocation env override (leaves launchers untouched):**

```bash
SFT_ADAPTER=/vol/checkpoints/sft_webshop_v3_mlpr32_${TS} \
    bash scripts/run_webshop_SOTA_attention_v1.sh
```

### Smoke (1 round × 5 eps) before committing the full RL budget

For each launcher, sanity-check that the rank-32 adapter loads without
shape errors and that R0 eval pct_success matches Phase 4's SFT eval
(±5 pp):

```bash
for launcher in attention flatGRPO LLMJudge; do
  ROUNDS=1 EPS_PER_ROUND=5 EVAL_EPS=20 \
    bash scripts/run_webshop_SOTA_${launcher}_v1.sh
done
```

Cost ~$10, ~30 min total. If any launcher errors at
`policy.load_adapter` (`Missing/Unexpected key(s)`), the rank-32 + MLP
config doesn't match the adapter — halt and re-check Phase 3
(`infra/app_sft_train.py` should print `LoRA arch: rank=32 target_modules=[...7 modules...]`).

---

## Phase 6 — launch the 4 RL methods (parallel-safe)

Each launcher backgrounds via `nohup` + writes its own PID/log file.
All 4 are parallel-safe (disjoint task ranges, disjoint cache dirs,
disjoint run-name prefixes).

```bash
bash scripts/run_webshop_SOTA_attention_v1.sh
bash scripts/run_webshop_SOTA_flatGRPO_v1.sh
bash scripts/run_webshop_SOTA_LLMJudge_v1.sh
bash scripts/run_webshop_SOTA_Progress_v1.sh
```

| Launcher        | Recipe                                       | Seed | Train range       | Eval range     | Cost / wall-clock |
|-----------------|----------------------------------------------|-----:|-------------------|----------------|-------------------|
| attention v1    | TurnRD-V2 (turn-level decomposer), α=0.5     | 11   | [8800, 9600)      | [6500, 6600)   | ~$20 / 3-4 hr     |
| flatGRPO v1     | Flat GRPO + attribute-progress dense signal  | 23   | [18400, 19200)    | [6500, 6600)   | ~$15 / 2.5-3 hr   |
| LLMJudge v1     | LLM judge (OpenAI gpt-4o-mini), α=0.5        | 31   | [24800, 25600)    | [6500, 6600)   | ~$25 / 3 hr       |
| Progress v1     | Method C: decomposer=progress, α=0.5         | 41   | [32800, 33600)    | [6500, 6600)   | ~$15 / 2.5-3 hr   |

Tunable per-launcher (env vars, defaults shown):

```bash
ROUNDS=10  START_ROUND=0  EPS_PER_ROUND=80  EVAL_EPS=100
SEED=<launcher-specific>  ROLLOUT_TEMP=1.0  TURNRD_EPOCHS=5
SFT_ADAPTER=<set in Phase 5>
```

### Monitor

```bash
tail -f /tmp/webshop_attention_v1.log
tail -f /tmp/webshop_flatGRPO_v1.log
tail -f /tmp/webshop_LLMJudge_v1.log

# Or pull the per-round eval json directly:
modal volume get cs224r-hgpo-vol \
    /manifests/webshop_attention_v1_R9_*/eval_log.json eval_R9.json
```

### Pass criterion

Final-round (R9) eval pct_success ≥ 0.40 on `[6500, 6600)` (current
0.28 baseline + 12 pp lift). The rank-32 + MLP SFT warm-start is the
sole moving variable across this bake-off vs the previous rank-16
baseline.

---

## Repo map (relevant files only)

```
configs/
  env_webshop.json                              — base env spec
  SFTOnly_webshop_mlpr32.json                   — eval-only spec, rank-32+MLP policy
  TurnRDV2_webshop_SOTA_10round_mlpr32_v1.json  — attention RL config
  flatGRPO_webshop_SOTA_10round_mlpr32_v1.json  — flat GRPO RL config
  LLMJudge_webshop_SOTA_10round_mlpr32_v1.json  — LLM judge RL config
infra/
  app_webshop_install.py    — Phase 0: install env + spaCy + BM25 index
  app_data.py               — optional: download upstream gdown human trajs
  app_webshop_sft_gen.py    — Phase 1: oracle SFT trajectory generator
  app_sft_train.py          — Phase 3: SFT trainer (LoRA arch CLI-plumbed)
  app_train_loop.py         — Phase 4 (eval) + Phase 6 (RL train loop)
scripts/
  run_webshop_sft_v3_mlpr32.sh   — Phases 1-4 launcher (4-step pipeline)
  run_webshop_SOTA_attention_v1.sh — Phase 6: attention RL launcher
  run_webshop_SOTA_flatGRPO_v1.sh  — Phase 6: flat GRPO RL launcher
  run_webshop_SOTA_LLMJudge_v1.sh  — Phase 6: LLM judge RL launcher
  run_webshop_SOTA_Progress_v1.sh  — Phase 6: HGPO-Progress RL launcher
src/datasets/
  sft_webshop.py            — both loaders + ReAct prompt renderer
src/envs/
  webshop_adapter.py        — env adapter + goal-introspection helpers
  prompts/react_webshop.py  — single-source-of-truth ReAct prompt template
tests/unit/
  test_sft_webshop_loader.py — 30 unit tests (URL parsing, loaders, prompt parity)
```

---

## Cost / time budget

| Phase                            | Cost      | Wall-clock      |
|----------------------------------|----------:|----------------:|
| 0: WebShop env install           | ~$1       | ~20-30 min      |
| 0: gdown human-trajs download    | ~$0.10    | ~5 min          |
| 1: Oracle gen (2000 sessions)    | ~$0.50    | ~30-60 min      |
| 3: SFT train (6 ep × ~15k ex)    | ~$15-30   | ~3-5 hr         |
| 4: SFT eval (200 eps)            | ~$3-5     | ~30 min         |
| 5: R0 smoke × 3 RL configs       | ~$10      | ~30 min         |
| 6: 4 RL methods × 10 rounds      | ~$75      | ~3-4 hr each    |
| **Total**                        | **~$90**  | **~6-8 hr**     |

(RL methods run in parallel so wall-clock is bounded by the slowest.)

---

## Troubleshooting

- **`policy.load_adapter` raises `Missing key(s)`** at R0 of an RL
  launcher → the SFT adapter was trained at rank-16 (or different
  target modules) than the RL config expects. Verify Phase 3's log
  shows `LoRA arch: rank=32 target_modules=['q_proj', ..., 'down_proj']`
  (7 modules total). Re-run Phase 3 with `LORA_RANK=32` if not.
- **Gen win rate < 50 %** → the upstream WebShop env may have been
  re-indexed with a different product split. Re-run Phase 0's
  `build_index_1k` action. Alternatively soften the oracle by passing
  `REWARD_THRESHOLD=0.5` (degrades from "exact attribute match" to
  "≥ half attrs matched").
- **SFT eval pct_success < 0.42** → the rank-32 + MLP arch is
  over-parameterizing on the 15k corpus. Re-run Phase 3 with
  `EPOCHS=3` (the 6-ep × 5e-5 recipe is tuned for ~10k AlfWorld
  examples).
- **LLMJudge RL fails with `Secret 'openai-secret' not found`** →
  either provision the secret (`modal secret create openai-secret
  OPENAI_API_KEY=sk-...`) or skip LLMJudge (`CS224R_SKIP_OPENAI_SECRET=1`
  + only launch attention + flatGRPO).
- **`modal run` hangs at "Loading image"** → first invocation triggers
  a ~5-10 min image build; subsequent runs reuse the cached image.
- **Volume disk full** → `modal volume rm cs224r-hgpo-vol /checkpoints/sft_v1_*`
  to prune old adapters. Each adapter is ~50-100 MB.

---

## References

- Plan file: `~/.llms/plans/webshop_sft_mlpr32_oracle_baseline.plan.md`
- Sibling AlfWorld pipeline (same recipe shape):
  `scripts/run_alfworld_sft_v2.sh` + `infra/app_sft_train_alfworld.py`
