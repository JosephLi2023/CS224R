# Data & Artifact Inventory — for Final-Report Analysis

> **Purpose:** complete index of every Modal volume artifact and local log produced by the AlfWorld goalcondFiLM SOTA campaign, plus schema docs + analysis entry points. Use this as the canonical reference when running post-hoc analysis (FiLM γ/β trajectories, eval curves, per-task-type breakdowns, dead-K visualizations, advantage-signal plots, replay-buffer-derived stats) for the final report.

---

## TL;DR — primary artifacts to use

| What you want | Where to find it |
|---|---|
| **SOTA LoRA adapter (the 80% policy)** | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round13_adapter/` on `cs224r-hgpo-vol` |
| **SOTA TurnRD V-head + FiLM ckpt** | `/vol/cache/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM/ckpt.pt` (currently R12-vintage post-cleanup) |
| **SOTA replay buffer (all R0-R14 rows)** | `/vol/cache/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM/replay.jsonl` |
| **Per-round training metrics** | `/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round{NN}_<ts>/train_log.json` |
| **Prior baseline (R12=73%, plain v3)** | `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round12_adapter/` and `/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/{ckpt.pt,replay.jsonl}` |
| **Local orchestrator logs (full stdout)** | `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM*.log` (see local logs section) |

---

## Experiment lineage map

The goalcondFiLM SOTA was 4 chained orchestrator launches against one shared prefix. The prior baseline is a separate prefix.

```
PRIOR BASELINE (prefix: TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31)
└── R0-R12 (K=8 plain v3, no FiLM, no goal-cond)              → R12 eval = 0.73

SOTA LINEAGE (prefix: TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31)
├── Phase A: R0-R4   K=8  goalcondFiLM base 5-round            → R4 eval = 0.70
├── Phase B: R5-R9   K=8  extend                               → R9 eval = 0.75
├── Phase C: R10-R12 K=12 extend (K bump)                      → R12 eval = 0.79
├── [DEPRECATED] hard-filter R13-R15 (deleted; logs archived)  → regressed to 0.77/0.77/0.75
│   └── Cleanup performed: deleted Modal artifacts, trimmed replay R13-R15 rows,
│       re-ran train_turnrd to regen ckpt.pt at R12-vintage state
└── Phase D: R13-R14 K=12 unfiltered relaunch                  → R13 eval = 0.80 (SOTA), R14 = 0.75 (auto-killed)
```

---

## Modal volume artifacts — full inventory

All artifacts on Modal volume **`cs224r-hgpo-vol`** under environment `main`. Access via:
```bash
modal volume ls cs224r-hgpo-vol <path>
modal volume get cs224r-hgpo-vol <src> <local_dst> [--force]
```

### A. SOTA goalcondFiLM lineage (seed-31)

**Cache dir** (replay + standalone TurnRD ckpt):
```
/vol/cache/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM/
├── replay.jsonl   ~2.6 GB, ~9,376 R0-R12 rows + ~1,920 R13-R14 rows (~11,296 total)
└── ckpt.pt        ~2.4 MB, R12-vintage TurnRDv2 + FiLM (regenerated post-cleanup)
```

**LoRA adapters** (carry-policy chain, R0 → R14):
```
/vol/checkpoints/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round{00..14}_adapter/
├── adapter_model.safetensors
├── adapter_config.json
└── README.md
```
**SOTA adapter**: `..._round13_adapter` ← the 0.80 policy.

**Per-round manifests** (one dir per round, includes `train_log.json` + Modal stdout fragments):
```
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round00_20260528_183644
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round01_20260528_195654
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round02_20260528_211052
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round03_20260528_223021
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round04_20260528_234441
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round05_20260529_014419
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round06_20260529_030317
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round07_20260529_043535
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round08_20260529_054952
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round09_20260529_065139
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round10_20260529_202035
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round11_20260529_215927
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round12_20260530_004647    ← R12 (1st attempt; eval may be partial)
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round12_20260530_010029    ← R12 (final; use this one for R12 metrics)
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round13_20260530_155323    ← R13 SOTA round (use for SOTA metrics)
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round14_20260530_184544    ← R14 (1st attempt; eval partial)
/vol/manifests/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round14_20260530_185742    ← R14 (final; auto-killed at 0.75)
```
When a round has multiple timestamped dirs (e.g. R12, R14), the **lex-largest timestamp = the authoritative one**. Earlier ones are partial dumps from retries.

### B. Prior baseline (plain v3, no FiLM, no goal-cond)

**Cache dir**:
```
/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3/
├── replay.jsonl   (K=8 rows from R0-R12; no goal_emb/goal_text fields)
└── ckpt.pt        (TurnRDv2 without goal_proj/gamma/beta keys)
```

**LoRA adapters** (R0-R12):
```
/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round{00..12}_adapter/
```

**Manifests** (R0-R12):
```
/vol/manifests/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_seed31_round{00..12}_<ts>/train_log.json
```
Authoritative timestamps: R0=20260522_231632 ... R12=20260523_181258 (see prior `sota_R12.md` for the full list).

### C. SFT warm-start adapter (used for both R0s)

```
/vol/checkpoints/sft_alfworld_v2_e3_20260521_145134_20260521_145149/
```
Rank-32 LoRA MLP+attn SFT on Qwen2.5-1.5B-Instruct. Same adapter for both the v3 baseline R0 and the goalcondFiLM R0.

### D. Other cache dirs on the volume (NOT used for SOTA — flagged for cleanup awareness)

| Cache dir | Status | Note |
|---|---|---|
| `cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalaware_replay` | DEPRECATED | early goal-aware experiment |
| `cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalcond` | DEPRECATED | pre-FiLM goalcond V1 (the one with per-turn supervision that regressed) |
| `cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalsup_replay` | DEPRECATED | goalsup experiment replay |
| `cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalsup_replay_validseen` | DEPRECATED | goalsup eval replay |
| `cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget` | SEPARATE EXPERIMENT | xlbudget run, unrelated to SOTA |

These can be ignored for final-report analysis. Free up volume space later if needed.

---

## Local log files (orchestrator stdout, archived)

All under `/tmp/`. These are the **full local stdout** of each orchestrator (one log per phase). Contain modal-run subprocess commands, eval results, Modal app IDs, and (occasionally) flush'd round transitions.

| Local log path | Phase | Size | Status |
|---|---|---|---|
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.BROKEN_NO_FILM.log` | Original 5-round attempt before the `--goal-conditioned-value-head` orchestrator fix | 7.4 MB | Forensic — FiLM ckpts had no goal_proj/gamma/beta keys; superseded by re-run |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.PREEMPTED.log` | 5-round re-launch that got Modal-preempted at R0 ep 40 | 0 B (empty) | Forensic — empty because Python stdout buffering didn't flush before kill |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM.log` | Phase A: R0-R4 K=8 base goalcondFiLM (the run that finally landed cleanly) | 7.3 MB | Use for R0-R4 metrics |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_extend_R5R9.log` | Phase B: R5-R9 K=8 extend | 6.4 MB | Use for R5-R9 metrics |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_extend_R10R12_K12.log` | Phase C: R10-R12 K=12 extend | 3.2 MB | Use for R10-R12 metrics |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_extend_R13R15_K12_hardfilter.R13_HANG.log` | DEPRECATED hard-filter attempt (R13 hang at sync_every=12; replay corrupted vLLM) | 0 B | Forensic — empty (buffering) |
| `/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_extend_R13R15_K12_hardfilter.log` | DEPRECATED hard-filter R13-R15 (the one that produced R13=77/R14=77/R15=75 regression) | 3.1 MB | Forensic — captures the regression we then reverted |
| **`/tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_extend_R13R15_K12_unfiltered.log`** | **Phase D: R13-R14 K=12 unfiltered relaunch (the SOTA run)** | 3.1 MB | **Use for R13 SOTA metrics** |

### Recommended action: archive logs to repo for permanence
The `/tmp/` files will eventually be wiped on host reboot. Consider:
```bash
mkdir -p reports/milestone/logs_sota_R13/
cp /tmp/turnrd_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM*.log reports/milestone/logs_sota_R13/
```

---

## Schema documentation for each artifact type

### 1. `train_log.json` (per round, in manifest dir)

Top-level keys:
```python
{
  "rows": [             # one dict per rollout episode
    {
      "episode": int,           # 0..n_episodes-1
      "task_id": int,           # AlfWorld task id
      "mean_reward": float,     # mean across K trajectories
      "std_reward": float,      # std across K trajectories (used for dead-K detection)
      "completed": int,         # how many of K trajectories reached terminal success
      "truncated": int,         # how many hit max_turns without success
      "n_turns": int,           # total turn count across K
      "n_action_tokens": int,
      "policy_loss": float,     # H-GRPO policy gradient loss
      "kl_term": float,
      "consistency": float,
      "consistency_t": float,   # turn-level consistency
      "total_loss": float,
      "observed_kl": float,
      "kl_coef": float,
      "grad_norm": float,
      "turnrd_grad_norm": float,
      "mean_traj_adv": float,   # mean trajectory advantage (H-GRPO)
      "mean_turn_adv": float,   # mean per-turn advantage (TurnRD-decomposed)
      "cls_query_norm": float,
      "alpha_mean": float,      # mean α (TurnRD attention over turns)
      "alpha_var": float,
      "alpha_max": float,
      "alpha_entropy": float,   # softmax entropy of α distribution
      "alpha_progress_corr": float,  # correlation of α with progress signal
      "std_reward_group": float,
      "dead_K_group": int,      # 1 if std_reward < 1e-12 (no policy gradient), else 0
      "mean_abs_traj_adv": float,
      "std_traj_adv": float,
      "mean_abs_adv_token": float,
      "elapsed_s": float,       # wall time for this episode (rollout + train step)
      "error": str,             # only if episode crashed
      "traceback": str          # full traceback if crashed (from app_train_loop.py)
    },
    ...
  ],
  "config": { ... },     # full config snapshot (n_episodes, K, max_turns, sync_every, ...)
  "eval": {              # appended after rollout+train, runs greedy K=1 on disjoint task slice
    "pct_success": float,      # the headline metric
    "n_episodes_attempted": int,
    "n_episodes_ok": int,
    "n_episodes_crashed": int,
    "K": 1,                    # eval always uses K=1 greedy
    ...
  },
  "early": [...],        # mean_reward of first 10% of rollout episodes (warmup detection)
  "late": [...],         # mean_reward of last 10% (convergence detection)
  "total_elapsed": float
}
```

### 2. `replay.jsonl` (cache dir, JSONL — one record per trajectory)

Schema per row:
```python
{
  "task_id": int,                  # AlfWorld task id
  "turn_embeds": [[float] * 1536]  # [T, 1536] — Qwen last-hidden-state at last token of each turn
                                   #            ← THE policy-dependent encoder input for TurnRD
  "final_reward": float,           # 0.0 (truncated) or 1.0 (success) for AlfWorld
  "progress": [float],             # [T] per-turn raw_env_reward
  "progress_signal": [float],      # [T] per-turn intermediate_reward (PDDL fact diff or expert plan delta)
                                   #     ← THE V-head's primary fitting target
  "judge_labels": null,            # None for Mode 1 (used only for Mode 2 judge-based supervision)
  "round_idx": int,                # which round emitted this row (drives recency decay)
  "goal_text": str,                # parsed "Your task is to: ..." string from env
  "goal_emb": [float] * 1536       # per-trajectory goal embedding (Qwen last-hidden of the goal text)
                                   #     ← THE FiLM modulation input
}
```
**Notes:**
- For the **prior baseline** (`..._mlpr32_v3` cache), `goal_text` and `goal_emb` keys are ABSENT (goalcondFiLM features were added later).
- The SOTA replay has been **trimmed** to R0-R12 + the R13-R14 K=12 unfiltered rows; R13-R15 hard-filter rows were removed during cleanup.
- Total ~11,296 rows = 6,400 (R0-R9 K=8) + 2,976 (R10-R12 K=12) + 1,920 (R13-R14 K=12).

### 3. `ckpt.pt` (cache dir, PyTorch state dict)

Standalone TurnRDv2 state dict. Use:
```python
import torch
sd = torch.load("/path/to/ckpt.pt", map_location="cpu", weights_only=True)
# 41 keys for goalcondFiLM ckpt, 35 keys for plain v3 ckpt
```

State-dict keys (goalcondFiLM 41-key version):
```
input_proj.{weight,bias}          # 1536 → 128 input projection
encoder.layers.{0,1}.*            # 2 transformer encoder layers (self_attn, linear1/2, norm1/2)
pos_embed.weight                  # 50 × 128 positional embeddings
score_head.{0,2}.{weight,bias}    # MLP 128→128→1 producing α scores
value_head.{0,2}.{weight,bias}    # MLP 128→128→1 producing per-turn V
goal_proj.{weight,bias}           # 1536 → 128 goal embedding projection (FiLM-only)
goal_gamma.{weight,bias}          # 128 → 128 FiLM γ (zero-init)
goal_beta.{weight,bias}           # 128 → 128 FiLM β (zero-init)
```
Plain v3 (35 keys) lacks `goal_proj/goal_gamma/goal_beta`.

**Re-instantiating the model from the ckpt:**
```python
from src.turnrd.model import TurnRDv2, TurnRDv2Config
model = TurnRDv2(
    TurnRDv2Config(
        n_layers=2, hidden_size=128, n_heads=4,
        max_turns=50, dropout=0.1, causal=False,
        progress_prior_strength=1.0,
        goal_conditioned_value_head=True,  # CRITICAL for FiLM ckpts
    ),
    input_dim=1536,
)
model.load_state_dict(sd, strict=False)
```

### 4. LoRA adapter dir (PEFT format)

Each `..._round{NN}_adapter/` contains:
- `adapter_model.safetensors` — LoRA weights (rank-32 on attn `q,k,v,o` + MLP `gate,up,down`)
- `adapter_config.json` — PEFT config (rank, alpha, target_modules)
- `README.md` — PEFT-generated metadata

**Loading for inference:**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
policy = PeftModel.from_pretrained(base, "/path/to/round13_adapter/")
```

---

## Analysis entry points (what to use for what)

| Analysis question | Data source | Tool / recipe |
|---|---|---|
| **Eval pct_success curve** | All `train_log.json` `eval.pct_success` fields | `scripts/monitor_goalcondFiLM_K12.py` already plots this; extract to PDF for report |
| **Dead-K rate per round** | `train_log.json` `rows[].dead_K_group` | Sum across rows, divide by `len(rows)`. Already in dashboard. |
| **FiLM γ/β growth trajectory** | Per-round ckpt downloads + state_dict norm comparison | Pattern in our earlier `monitor_goalcondFiLM_K12.py --ckpt`. We have only R0/R4/R9/R12/R15 cached locally; need to re-derive intermediates from snapshots if we want every round |
| **α / V-head learned distributions** | `train_log.json` `rows[].{alpha_mean, alpha_var, alpha_entropy, alpha_max, alpha_progress_corr}` | Time-series plot per round |
| **Per-turn advantage** | `train_log.json` `rows[].{mean_traj_adv, mean_turn_adv, std_traj_adv}` | Compare H-GRPO trajectory advantage vs TurnRD per-turn advantage; the "turn-level credit assignment" plot |
| **Replay buffer statistics** | `replay.jsonl` iterate rows | Count completion rates by task_id, turn-length distributions, goal_emb cluster analysis (PCA/UMAP), progress_signal histograms |
| **Per-task-type breakdown** | `replay.jsonl` `goal_text` strings → categorize (look_at, pick_and_place, clean, heat, cool, pick_two) | Group eval results by inferred task type. See `infra/app_alfworld_per_type_eval.py` for the existing eval-only entrypoint that does this |
| **Comparison: SOTA vs prior baseline** | Both `train_log.json` chains | Side-by-side eval curve; FiLM γ/β over time only meaningful for SOTA (baseline has no FiLM) |
| **Modal app logs** (if needed beyond train_log) | `modal app logs <app_id>` — Modal retains some logs after run completion | App IDs are in the orchestrator local logs (grep "View run at") |
| **Trajectory-level visualization** | `replay.jsonl` row + corresponding env state | Need to replay the policy in env using the LoRA adapter; see `scripts/render_turnrd_credits_latex.py` for an existing per-round LaTeX rendering recipe |
| **Specific R13 → R14 dip diagnostic** | R13 and R14 train_logs + adapters | Eval R13 adapter on multiple eval seeds (3-5 fresh n=100 evals) to bracket noise; R13 may actually be 78±3 |

---

## Suggested analysis cookbook for the report

### Recipe 1: Full eval curve PDF (priority — most informative single plot)

```bash
mkdir -p reports/figures/sota_R13
for r in $(seq 0 14); do
  # Pick the lex-largest timestamp manifest for each round
  dir=$(modal volume ls cs224r-hgpo-vol /manifests | grep "goalcondFiLM_seed31_round$(printf %02d $r)_" | tail -1 | awk '{print $1}')
  modal volume get cs224r-hgpo-vol "$dir/train_log.json" "reports/figures/sota_R13/R${r}_train_log.json" --force
done
# Then plot via matplotlib
```

### Recipe 2: FiLM γ/β every round

Cache `ckpt.pt` is OVERWRITTEN each round (the standalone trainer always writes back to the same path). We currently have:
- R0 (`/tmp/film_fixed_R0_ckpt/ckpt.pt`)
- R4 (`/tmp/film_fixed_R4_ckpt/ckpt.pt`)
- R9 (`/tmp/film_extend_R9_ckpt/ckpt.pt`)
- R15 (`/tmp/film_R15_ckpt/ckpt.pt`) ← from the deprecated hard-filter run
- Current (`/tmp/film_R12regen/ckpt.pt`) ← R12-vintage post-regen

For the report, these 4-5 snapshots are sufficient to show monotone γ/β growth. To get every round, would need to re-derive from replay (can re-run `train_turnrd` on prefix-of-replay with `--max-records N`).

### Recipe 3: Per-task-type eval at SOTA

Use `infra/app_alfworld_per_type_eval.py`:
```bash
modal run infra/app_alfworld_per_type_eval.py \
  --adapter /vol/checkpoints/TurnRDV2_alfworld_SOTA_5round_mlpr32_v3_goalcondFiLM_seed31_round13_adapter \
  --n-per-type 30 \
  --output-path /vol/manifests/sota_R13_per_type_eval.json
```
This addresses the "valid_seen pool only covers 2 of 6 task types" caveat from the SOTA doc.

### Recipe 4: SOTA-vs-baseline side-by-side (key report figure)

Use `train_log.json` from both lineages (goalcondFiLM and plain v3). Plot eval pct_success per round on the same axis. The plain v3 trajectory ends at R12=0.73; goalcondFiLM ends at R13=0.80. The +7pp delta is the headline visual.

---

## Costs to note for the writeup

| Compute item | Approximate cost | Wall-clock |
|---|---|---|
| Prior baseline (R0-R12 plain v3) | ~$55 | ~14 hr |
| goalcondFiLM Phase A (R0-R4) | ~$8 | ~3 hr |
| goalcondFiLM Phase B (R5-R9) | ~$10 | ~6 hr |
| goalcondFiLM Phase C (R10-R12) K=12 | ~$15 | ~3.5 hr |
| goalcondFiLM Phase D (R13-R14, the SOTA-producing run) | ~$15 | ~4 hr |
| Deprecated hard-filter (R13-R15) — sunk cost | ~$15 | ~6 hr |
| Deprecated BROKEN_NO_FILM original 5-round | ~$8 | ~3 hr |
| Cleanup ops (replay trim, ckpt regen) | ~$2 | ~10 min |
| **Total goalcondFiLM campaign** | **~$73** | **~25 hr** |
| **+Prior baseline shared** | **~$55** | **~14 hr** |
| **GRAND TOTAL** | **~$128** | **~39 hr** |

---

## Cross-references

- **SOTA write-up**: `reports/milestone/sota_R13_goalcondFiLM.md` — the headline result + reproducibility recipe
- **Prior baseline write-up**: `reports/milestone/sota_R12.md` — the v3 baseline (preserved for the +7pp delta narrative)
- **Banked next experiment**: `/Users/shoupeili/.llms/plans/turnrd_v2_continual_larger.plan.md` — cumulative training + larger architecture, deferred for future
- **Ablations narrative**: `reports/ablations/README.md` — discusses goalcond (now superseded), H-GRPO normalization bottleneck (still open)
- **Dashboard script**: `scripts/monitor_goalcondFiLM_K12.py` — reads from Modal volume, produces the eval/dead-K/FiLM table (can be adapted for any prefix)
