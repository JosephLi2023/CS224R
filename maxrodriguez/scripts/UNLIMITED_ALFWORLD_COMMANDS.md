# Unlimited ALFWorld SFT + Signed-Attention + GRPO Commands

Run these from the repo root after pulling the branch:

```powershell
cd C:\Users\maxlr\CS224R
git checkout codex/maxrodriguez-alfworld-preserve
git pull
$env:CS224R_SKIP_OPENAI_SECRET = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
$SFT_DATA = "/vol/data/alfworld/sft_trajs_maxrodriguez_runtime500x10_plus_seq2seq_structured.jsonl"
$TAG = "unlimited_$(Get-Date -Format yyyyMMdd_HHmmss)"

$SFT_NODAGGER = "/vol/checkpoints/maxrodriguez_unlimited/sft/$TAG/full_lr1e5_e10_nodagger"
$SFT_DAGGER = "/vol/checkpoints/maxrodriguez_unlimited/sft/$TAG/full_lr1e5_e10_dagger"
$SAT_DIR = "/vol/checkpoints/maxrodriguez_unlimited/signed_attention/$TAG/satf_alltrain_h512_l6_e8"
$SAT_CKPT = "$SAT_DIR/signed_attention_transformer.pt"
$GRPO_ROOT = "/vol/checkpoints/maxrodriguez_unlimited/grpo/$TAG"
```

Optional: if the full mixed SFT JSONL is missing on the Modal volume, regenerate a full train-split expert dataset first:

```powershell
modal run infra/app_alfworld_sft_gen.py `
  --action generate `
  --n-games 3553 `
  --output-path $SFT_DATA `
  --max-history-turns 3 `
  --max-steps-per-episode 80
```

## 1. Train Signed-Attention Transformer

This is the high-capacity signed turn-reward transformer: 6 layers, hidden size 512, 8 heads, trained on all 3553 ALFWorld train games with validation after every epoch. The trainer first tries `eval_in_distribution`; if that split exposes no handcoded expert plans, it falls back to a train-tail diagnostic MSE and records `validation_source` in `summary.json`.

```powershell
modal run maxrodriguez/grpo/app_signed_attention_transformer.py::train_signed_attention_transformer_model `
  --epochs 8 `
  --learning-rate 5e-5 `
  --hidden-size 512 `
  --n-layers 6 `
  --n-heads 8 `
  --dropout 0.05 `
  --train-trajectories 3553 `
  --val-trajectories 140 `
  --max-turns 30 `
  --seed 42 `
  --run-name "satf_${TAG}_alltrain_h512_l6_e8" `
  --output-dir $SAT_DIR `
  --base-model-path $BASE_MODEL `
  --validation-every-epochs 1
```

## 2. Train And Evaluate SFT, No DAgger

This trains on every loaded SFT row (`--max-examples 0`) for 10 epochs at the best full-data learning rate we have used so far (`1e-5`), validates CE each epoch, and runs full free-form greedy `valid_seen`/`valid_unseen` eval at the end.

```powershell
modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::train_sft_plus `
  --data-path $SFT_DATA `
  --epochs 10 `
  --learning-rate 1e-5 `
  --min-reward 1.0 `
  --max-seq-len 2048 `
  --micro-batch-size 1 `
  --grad-accum 8 `
  --log-every 100 `
  --seed 42 `
  --val-fraction 0.08 `
  --base-model-path $BASE_MODEL `
  --run-name "sft_${TAG}_full_lr1e5_e10_nodagger" `
  --output-dir $SFT_NODAGGER `
  --max-examples 0 `
  --full-finetune `
  --no-sample-after-load `
  --no-use-dagger `
  --dagger-episodes 0 `
  --dagger-max-turns 30 `
  --dagger-max-new-examples 0 `
  --dagger-mix-ratio 0.0 `
  --dagger-start-epoch 1 `
  --dagger-every-n-epochs 1 `
  --dagger-task-id-base 0 `
  --dagger-split train `
  --post-eval-seen-episodes 140 `
  --post-eval-unseen-episodes 134 `
  --post-eval-max-turns 30 `
  --post-eval-max-seq-len 2048 `
  --post-eval-task-id-base 0
```

## 3. Train And Evaluate SFT, With Full DAgger

This starts from the same supervised data, then after epoch 1 collects DAgger corrections across the full train split each epoch. The cap allows up to one additional dataset worth of DAgger rows, which is aggressive without letting corrections swamp the original expert corpus.

```powershell
modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::train_sft_plus `
  --data-path $SFT_DATA `
  --epochs 10 `
  --learning-rate 1e-5 `
  --min-reward 1.0 `
  --max-seq-len 2048 `
  --micro-batch-size 1 `
  --grad-accum 8 `
  --log-every 100 `
  --seed 42 `
  --val-fraction 0.08 `
  --base-model-path $BASE_MODEL `
  --run-name "sft_${TAG}_full_lr1e5_e10_dagger" `
  --output-dir $SFT_DAGGER `
  --max-examples 0 `
  --full-finetune `
  --no-sample-after-load `
  --use-dagger `
  --dagger-episodes 3553 `
  --dagger-max-turns 30 `
  --dagger-max-new-examples 200000 `
  --dagger-mix-ratio 1.0 `
  --dagger-start-epoch 1 `
  --dagger-every-n-epochs 1 `
  --dagger-task-id-base 0 `
  --dagger-split train `
  --post-eval-seen-episodes 140 `
  --post-eval-unseen-episodes 134 `
  --post-eval-max-turns 30 `
  --post-eval-max-seq-len 2048 `
  --post-eval-task-id-base 0
```

After the two SFT commands finish, read both summaries and choose exactly one SFT checkpoint for GRPO. The final GRPO comparison is controlled by one variable, `$SFT_CKPT`; every standard/variant GRPO command below uses that same selected checkpoint via `--sft-adapter $SFT_CKPT`. Do not run GRPO from both SFT checkpoints unless you intentionally want a separate 2x-cost ablation.

```powershell
modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::read_json_artifact `
  --path "$SFT_NODAGGER/summary.json"

modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::read_json_artifact `
  --path "$SFT_DAGGER/summary.json"

# Pick the checkpoint with higher post_train_eval seen/unseen success.
# Default to DAgger only if it actually wins or is tied on full greedy eval.
$SFT_CKPT = $SFT_DAGGER
# Use this instead if no-DAgger wins:
# $SFT_CKPT = $SFT_NODAGGER
```

## 4. Train Full-Data GRPO And Turn-Level Variants

These settings are the grid-supported settings from the subset sweep, scaled to all 3553 train games with `K=8`. The loop launches four runs from the one selected SFT checkpoint: standard `trajectory_only`, then `progress_delta`, `signed_attention`, and `admissible_margin`. Inline GRPO eval is disabled (`--eval-episodes 0`) because claims should use the full free-form seen/unseen eval commands below.

```powershell
$GRPO_METHODS = @(
  @{ method = "trajectory_only";    alpha = "0.0"; lr = "1e-6"; kl = "0.02" },
  @{ method = "progress_delta";     alpha = "0.1"; lr = "1e-6"; kl = "0.05" },
  @{ method = "signed_attention";   alpha = "0.1"; lr = "1e-6"; kl = "0.02" },
  @{ method = "admissible_margin";  alpha = "0.1"; lr = "1e-6"; kl = "0.05" }
)

foreach ($row in $GRPO_METHODS) {
  $method = $row.method
  modal run maxrodriguez/grpo/app_alfworld_grpo.py `
    --action launch_manual `
    --sft-adapter $SFT_CKPT `
    --alpha $($row.alpha) `
    --turn-reward-method $method `
    --learning-rate $($row.lr) `
    --kl-coeff $($row.kl) `
    --n-episodes 3553 `
    --k 8 `
    --max-turns 30 `
    --clip-eps 0.2 `
    --grad-accum-steps 1 `
    --max-tokens-per-microbatch 2048 `
    --kl-warmup-episodes 25 `
    --dataset-size-mode full `
    --eval-episodes 0 `
    --run-name-suffix "${TAG}_full3553_k8_a100" `
    --task-id-stride 37 `
    --signed-attention-transformer-ckpt $SAT_CKPT `
    --save-adapter-out "$GRPO_ROOT/$method"
}
```

## 5. Full Seen/Unseen Evaluation For GRPO Checkpoints

Run this after all GRPO commands finish. These are the only evaluation numbers to use for benchmark claims.

```powershell
foreach ($row in $GRPO_METHODS) {
  $method = $row.method
  $adapter = "$GRPO_ROOT/$method"

  modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy `
    --adapter-path $adapter `
    --checkpoint-type lora `
    --episodes 140 `
    --task-id-base 0 `
    --run-name "eval_${TAG}_${method}_seen140" `
    --max-turns 30 `
    --max-seq-len 2048 `
    --split eval_in_distribution

  modal run maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy `
    --adapter-path $adapter `
    --checkpoint-type lora `
    --episodes 134 `
    --task-id-base 0 `
    --run-name "eval_${TAG}_${method}_unseen134" `
    --max-turns 30 `
    --max-seq-len 2048 `
    --split eval_out_of_distribution
}
```

## Smoke Tests

These are cheap checks for CLI/config wiring only; they should finish before launching the expensive sequence.

```powershell
python -m py_compile `
  maxrodriguez/grpo/app_signed_attention_transformer.py `
  maxrodriguez/grpo/app_alfworld_grpo.py `
  maxrodriguez/grpo/turn_level_reward_methods_todo.py `
  maxrodriguez/supervised_FT/app_alfworld_sft_plus.py `
  src/trainers/train_hgpo.py `
  infra/app_train_loop.py

modal run maxrodriguez/grpo/app_alfworld_grpo.py `
  --action show_sample_config `
  --sft-adapter "/vol/checkpoints/smoke_sft" `
  --include-methods signed_attention `
  --signed-attention-transformer-ckpt "/vol/checkpoints/smoke_sat/signed_attention_transformer.pt"

modal run maxrodriguez/grpo/app_alfworld_grpo.py `
  --action show_manual_config `
  --sft-adapter "/vol/checkpoints/smoke_sft" `
  --alpha 0.0 `
  --turn-reward-method trajectory_only `
  --learning-rate 1e-6 `
  --kl-coeff 0.02 `
  --n-episodes 3553 `
  --k 8 `
  --max-turns 30 `
  --kl-warmup-episodes 25 `
  --dataset-size-mode full `
  --eval-episodes 0 `
  --run-name-suffix smoke_config_only `
  --task-id-stride 37 `
  --save-adapter-out "/vol/checkpoints/smoke_grpo"
```
