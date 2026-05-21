# GRPO Subset Sweep Compiled Results (2026-05-20)

## Best standard GRPO baseline

| Method | Alpha | LR | KL | Cheap subset success |
|---|---:|---:|---:|---:|
| `trajectory_only` | `0.0` | `1e-6` | `0.02` | `0.45` |

Note: `trajectory_only` with `lr=5e-6, kl=0.02` also reached `0.45`, but we selected `1e-6` as the cleaner winner.

## Best turn-level variants

| Variant | Alpha | LR | KL | Cheap subset success | Notes |
|---|---:|---:|---:|---:|---|
| `progress_delta` | `0.1` | `1e-6` | `0.05` | `0.45` | Best completed variant result |
| `signed_attention` | `0.1` | `1e-6` | `0.02` | `0.40` | `alpha = 0.1, 0.5, 0.9` all tied at `0.40`; chose simplest tied winner |
| `admissible_margin` | `0.1` | `1e-6` | `0.05` | `0.45` | Tied the best observed cheap subset score |

## Variant alpha comparison at the chosen LR/KL

### `progress_delta` at `lr=1e-6`, `kl=0.05`

| Alpha | Cheap subset success |
|---:|---:|
| `0.1` | `0.45` |
| `0.5` | `0.40` |
| `0.9` | `0.40` |

### `signed_attention` at `lr=1e-6`, `kl=0.02`

| Alpha | Cheap subset success |
|---:|---:|
| `0.1` | `0.40` |
| `0.5` | `0.40` |
| `0.9` | `0.40` |

### `admissible_margin` at `lr=1e-6`, `kl=0.05`

| Alpha | Cheap subset success |
|---:|---:|
| `0.1` | `0.45` |
| `0.5` | `0.40` |
| `0.9` | `0.40` |

## Full-data final runs launched

These are the 4 full-data GRPO finals currently launched:

| Category | Run name |
|---|---|
| Standard GRPO baseline | `max_grpo_trajectory_only_a0_lr1em06_kl0p02_k4_t30_full_milestone_v12_fullfinals` |
| `progress_delta` final | `max_grpo_progress_delta_a0p1_lr1em06_kl0p05_k4_t30_full_milestone_v12_fullfinals` |
| `signed_attention` final | `max_grpo_signed_attention_a0p1_lr1em06_kl0p02_k4_t30_full_milestone_v12_fullfinals` |
| `admissible_margin` final | `max_grpo_admissible_margin_a0p1_lr1em06_kl0p05_k4_t30_full_milestone_v12_fullfinals` |

All 4 use:

- full ALFWorld train split (`3553` train tasks)
- full seen eval split (`140` tasks) configured in the ALFWorld data paths
- full unseen eval split (`134` tasks) configured in the ALFWorld data paths
- best final SFT warm start:
  `/vol/checkpoints/maxrodriguez_milestone/final_sft/sftfinal_milestone_v8_e3_lr2E-05_seq1024_ga4_nodagger`

The explicit full seen/unseen benchmark eval stage will be run after these 4 full-data GRPO trainings finish and write checkpoints.
