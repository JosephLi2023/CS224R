# AlfWorld SOTA — R12 / 73 % pct_success

> **One-line summary:** the policy reached **73 % success (n=100, valid_seen-pool eval)** at **Round 12** of the v3 SoTA training schedule, using attention-based H-GRPO with TurnRDv2 credit decomposition + recency-decay buffer + carry-policy across rounds. This is the headline number used in the milestone report.

---

## Round-by-round eval trajectory

Source: `/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.log` (extend launcher) and `/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3.log` (R0–R9).

| Round | Eval pct_success (n=100, task_id_base=6500) | log line |
|---:|---:|---:|
| R0 | 0.62 | (v3 main log) |
| R6 | 0.70 | (v3 main log) |
| R9 | 0.71 | (v3 main log) |
| **R10** | **0.69** | extend log line 12853 |
| **R11** | **0.71** | extend log line 24493 |
| **R12** | **0.73** ← **headline** | extend log line 39190 |

The R12 eval line: `>>> Eval done: avg_R=0.7300 (±0.4440) | pct_success=0.730 | ok=100/100 | elapsed=496.44s`

---

## Provenance

### Config file

**`/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3.json`**

Key knobs (full file authoritative; cited here for the report):

| Block | Knob | Value |
|---|---|---|
| `run` | seed | 31 |
| `env` | task_split | train |
| `env` | num_train_games | 400 |
| `env` | num_eval_games | 50 |
| `env` | use_textworld_intermediate_reward | true |
| `env` | use_facts_diff_intermediate_reward | true |
| `policy` | lora_rank / lora_alpha | 32 / 64 |
| `policy` | lora_target_modules | `q,k,v,o,gate,up,down` (attn + MLP) |
| `turnrd` | version | v2 (attention-based decomposer) |
| `turnrd` | layers / hidden_size / n_heads | 2 / 128 / 4 |
| `turnrd` | causal | false (bidirectional) |
| `turnrd` | lambda_value / lambda_rank / lambda_progress | 1.0 / 0.1 / 0.01 |
| `turnrd` | recency_decay_half_life | 4 rounds |
| `trainer` | K (rollouts per group) | 8 |
| `trainer` | rollout_temperature | 1.0 (T=1.0 for diversity) |
| `trainer` | carry_policy_across_rounds | true |

### Launcher scripts

**Phase A (R0–R9, the original 10-round schedule):**
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/scripts/run_alfworld_SOTA_10round_mlpr32_v3.sh`
- Defaults: `ROUNDS=10`, `START_ROUND=0`, `SEED=31`, `EPS_PER_ROUND=80`, `EVAL_EPS=100`, `TURNRD_EPOCHS=5`.

**Phase B (R10–R12, the extension that produced 0.73):**
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/scripts/run_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.sh`
- Defaults: `ROUNDS=13`, `START_ROUND=10`, same SEED=31, same EPS=80, same EVAL=100, same TURNRD_EPOCHS=5.
- Reuses the same config + same replay/ckpt paths so the recency-decay buffer and TurnRD V-head ckpt carry forward seamlessly.

**Warm-start adapter for R0:**
- `/vol/checkpoints/sft_alfworld_v2_e3_20260521_140452_20260521_140503` (or `..._145134_...`) — rank-32 LoRA MLP+attn SFT on Qwen2.5-1.5B-Instruct.

### Launch command (for reproducibility)

```bash
# Phase A: R0 -> R9
SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_140452_20260521_140503 \
  bash scripts/run_alfworld_SOTA_10round_mlpr32_v3.sh

# Phase B: R10 -> R12 (after Phase A completed)
SFT_ADAPTER=/vol/checkpoints/sft_alfworld_v2_e3_20260521_140452_20260521_140503 \
  bash scripts/run_alfworld_SOTA_10round_mlpr32_v3_extend_R10R12.sh
```

### Modal volume artifacts (all on `cs224r-hgpo-vol`)

**Per-round LoRA adapters** (carry-policy: each round's adapter is the warm-start for the next):

| Round | Path on volume |
|---:|---|
| 00 | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round00_adapter` |
| 09 | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round09_adapter` |
| 10 | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round10_adapter` |
| 11 | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round11_adapter` |
| **12** | **`/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round12_adapter`** ← **the 0.73 policy** |

**Replay buffer (cumulative across R0–R12, with recency-decay applied at train time):**
- `/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/replay.jsonl`

**TurnRDv2 V-head checkpoint (cumulative, carried across all 13 rounds):**
- `/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/ckpt.pt`

**Modal Apps (web logs available):**
- R12 train_loop: `ap-NMHth2FrMKgSIFUaFG0KWQ` (extend log line 39194)
- R12 train_turnrd: `ap-O7PIB3Oc0HOZP1unaWDQgE` (extend log line 39347)

---

## Citation snippet for the report

> Our best AlfWorld policy reaches **73 % success rate** on the AlfWorld valid_seen pool (n=100, task_id_base=6500, greedy decoding) at the end of **Round 12** of training. The model is a Qwen2.5-1.5B-Instruct base with a rank-32 LoRA adapter targeting both attention and MLP projections, fine-tuned via H-GRPO with an attention-based TurnRDv2 credit decomposer and α-blend coefficient `lambda_value=1.0`, with per-batch recency decay (half-life=4 rounds) on the cumulative replay buffer. Training used 80 episodes per round × 13 rounds × K=8 rollouts per group, totalling 8,320 trajectories at rollout temperature T=1.0. Per-round eval was n=100 trajectories on a held-out task ID slice (IDs 6500–6599) disjoint from the training pool. Total Modal compute: ~14 hours on an A100-80GB, ~$55. The full launch recipe (configs, scripts, checkpoint paths) is captured in `reports/milestone/sota_R12.md`.

---

## Per-type caveat (important for the report)

The 73 % number is on the **valid_seen pool**, which due to AlfWorld's upstream alphabetical-first-chunk-of-N bias contains **only `look_at_obj_in_light` and `pick_and_place_simple` task types** (the two alphabetically-first of six). The other four task types (`clean`, `heat`, `cool`, `pick_two`) are **not represented in this eval slice** — see `reports/turnrd_credit_assignment_demo/per_type_eval/breakdown.txt` for the per-type analysis. **Per-type breakdown for hard task types is the open follow-up** addressed by the in-progress goal-aware-supervision run (plan `turnrd_goalsup_rl_loop_integration`).
