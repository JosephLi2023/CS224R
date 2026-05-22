param(
  [ValidateSet("sft-grid", "sft-final", "sft-final-eval", "sft-dagger-comparison", "signed-attention-grid", "signed-attention-final", "grpo-grid", "grpo-grid-core", "grpo-grid-signed-attention", "grpo-final", "grpo-final-eval", "collect", "all")]
  [string]$Stage = "all",
  [string]$RunTag = "milestone",
  [string]$BaseModelPath = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$DataPath = "/vol/data/alfworld/sft_trajs_maxrodriguez_500.jsonl",
  [double]$BestLearningRate = 0.0,
  [int]$BestMaxSeqLen = 2048,
  [int]$BestGradAccum = 4,
  [string]$BestSftCheckpoint = "",
  [string]$SignedAttentionTransformerCheckpoint = "",
  [double]$BestSignedAttentionLearningRate = 0.0,
  [int]$BestSignedAttentionHiddenSize = 128,
  [int]$BestSignedAttentionNumLayers = 2,
  [int]$BestSignedAttentionNumHeads = 4,
  [double]$BestSignedAttentionDropout = 0.0,
  [int]$BestSignedAttentionTrainTrajectories = 256,
  [int]$BestSignedAttentionValTrajectories = 64,
  [int]$BestSignedAttentionMaxTurns = 30,
  [string]$FinalEvalCheckpointPath = "",
  [ValidateSet("full", "lora")]
  [string]$FinalEvalCheckpointType = "full",
  [int]$SftGridEvalEpisodes = 20,
  [int]$GrpoGridEvalEpisodes = 20,
  [int]$GrpoMaxSpecs = 0,
  [string]$FinalGrpoMethod = "",
  [double]$FinalGrpoAlpha = -1.0,
  [double]$FinalGrpoLearningRate = 0.0,
  [double]$FinalGrpoKlCoeff = 0.0,
  [int]$FinalGrpoEpisodes = 100,
  [int]$FinalGrpoK = 4,
  [int]$FinalGrpoMaxTurns = 30,
  [double]$FinalGrpoClipEps = 0.2,
  [int]$FinalGrpoGradAccumSteps = 1,
  [int]$FinalGrpoMaxTokensPerMicrobatch = 2048,
  [int]$FinalGrpoKlWarmupEpisodes = 5,
  [int]$FinalGrpoTaskIdStride = 37,
  [string]$FinalGrpoDatasetSizeMode = "full",
  [int]$FinalGrpoEvalEpisodes = 0,
  [int]$FullSeenEpisodes = 140,
  [int]$FullUnseenEpisodes = 134,
  [int]$SftPostEvalSeenEpisodes = 0,
  [int]$SftPostEvalUnseenEpisodes = 0,
  [string]$DownloadDir = "maxrodriguez/results/modal_manifests",
  [string]$LedgerPath = "maxrodriguez/results/milestone_launch_ledger.jsonl"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$stateRoot = Join-Path $env:LOCALAPPDATA "CodexModalMilestone"
$launchRoot = Join-Path $stateRoot $RunTag
$submitDir = Join-Path $launchRoot "submit_logs"
$launchLedgerPath = Join-Path $launchRoot "milestone_launch_ledger.jsonl"

New-Item -ItemType Directory -Force -Path $launchRoot | Out-Null
New-Item -ItemType Directory -Force -Path $submitDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $LedgerPath) | Out-Null

function Write-Ledger {
  param([string]$RunName, [string]$Kind, [string[]]$ModalArgs)
  $row = [ordered]@{
    launched_at = (Get-Date).ToUniversalTime().ToString("o")
    run_name = $RunName
    kind = $Kind
    command = "modal " + (($ModalArgs | ForEach-Object { "$_" }) -join " ")
  }
  $json = $row | ConvertTo-Json -Compress
  Add-Content -Path $launchLedgerPath -Value $json
  Add-Content -Path $LedgerPath -Value $json
}

function Invoke-DetachedModal {
  param([string]$RunName, [string]$Kind, [string[]]$ModalArgs)
  Write-Host "Launching [$Kind] $RunName"
  Write-Ledger -RunName $RunName -Kind $Kind -ModalArgs $ModalArgs
  $stdoutPath = Join-Path $submitDir "${RunName}.out.log"
  $stderrPath = Join-Path $submitDir "${RunName}.err.log"
  Start-Process -FilePath "modal" `
    -ArgumentList $ModalArgs `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden | Out-Null
}

function Add-DaggerArgs {
  param([string[]]$BaseArgs, [bool]$UseDagger)
  if ($UseDagger) {
    return $BaseArgs + @(
      "--use-dagger",
      "--dagger-episodes", "5",
      "--dagger-max-turns", "10",
      "--dagger-max-new-examples", "100",
      "--dagger-mix-ratio", "0.1",
      "--dagger-start-epoch", "0",
      "--dagger-every-n-epochs", "2",
      "--dagger-task-id-base", "8000",
      "--dagger-split", "train"
    )
  }
  return $BaseArgs + @(
    "--no-use-dagger",
    "--dagger-episodes", "0",
    "--dagger-max-turns", "10",
    "--dagger-max-new-examples", "0",
    "--dagger-mix-ratio", "0.0",
    "--dagger-start-epoch", "0",
    "--dagger-every-n-epochs", "1",
    "--dagger-task-id-base", "8000",
    "--dagger-split", "train"
  )
}

function Launch-SftGrid {
  $learningRates = @("0.0001", "0.00002", "0.00001", "0.000001")
  $maxSeqLens = @(1024, 2048)
  $gradAccums = @(4, 8)

  foreach ($lr in $learningRates) {
    foreach ($seqLen in $maxSeqLens) {
      foreach ($gradAccum in $gradAccums) {
        $runName = "sftgrid_${RunTag}_e1_lr$($lr.Replace('.', 'p'))_seq${seqLen}_ga${gradAccum}_nodagger"
        $outputDir = "/vol/checkpoints/maxrodriguez_milestone/sft_grid/$runName"
        $modalArgs = @(
          "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::train_sft_plus",
          "--data-path", $DataPath,
          "--epochs", "1",
          "--learning-rate", $lr,
          "--min-reward", "1.0",
          "--max-seq-len", "$seqLen",
          "--micro-batch-size", "1",
          "--grad-accum", "$gradAccum",
          "--log-every", "25",
          "--seed", "42",
          "--val-fraction", "0.08",
          "--base-model-path", $BaseModelPath,
          "--run-name", $runName,
          "--output-dir", $outputDir,
          "--max-examples", "0"
        )
        $modalArgs = Add-DaggerArgs -BaseArgs $modalArgs -UseDagger $false
        Invoke-DetachedModal -RunName $runName -Kind "sft-grid" -ModalArgs $modalArgs
      }
    }
  }
}

function Launch-FinalSft {
  param([bool]$UseDagger)
  if ($BestLearningRate -le 0.0) {
    throw "Set -BestLearningRate from the winning 1-epoch SFT grid run."
  }
  $daggerTag = if ($UseDagger) { "dagger" } else { "nodagger" }
  $runName = "sftfinal_${RunTag}_e3_lr$($BestLearningRate.ToString().Replace('.', 'p'))_seq${BestMaxSeqLen}_ga${BestGradAccum}_${daggerTag}"
  $outputDir = "/vol/checkpoints/maxrodriguez_milestone/final_sft/$runName"
  $modalArgs = @(
    "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::train_sft_plus",
    "--data-path", $DataPath,
    "--epochs", "3",
    "--learning-rate", "$BestLearningRate",
    "--min-reward", "1.0",
    "--max-seq-len", "$BestMaxSeqLen",
    "--micro-batch-size", "1",
    "--grad-accum", "$BestGradAccum",
    "--log-every", "25",
    "--seed", "42",
    "--val-fraction", "0.08",
    "--base-model-path", $BaseModelPath,
    "--run-name", $runName,
    "--output-dir", $outputDir,
    "--max-examples", "0",
    "--post-eval-seen-episodes", "$SftPostEvalSeenEpisodes",
    "--post-eval-unseen-episodes", "$SftPostEvalUnseenEpisodes",
    "--post-eval-max-turns", "30",
    "--post-eval-max-seq-len", "$BestMaxSeqLen",
    "--post-eval-task-id-base", "0"
  )
  $modalArgs = Add-DaggerArgs -BaseArgs $modalArgs -UseDagger $UseDagger
  Invoke-DetachedModal -RunName $runName -Kind "sft-final" -ModalArgs $modalArgs
  Write-Host "Final SFT checkpoint path: $outputDir"
}

function Launch-FinalSftEval {
  if (-not $FinalEvalCheckpointPath) {
    throw "Set -FinalEvalCheckpointPath to the final SFT checkpoint to benchmark."
  }

  $seenRunName = "sfteval_${RunTag}_seen${FullSeenEpisodes}"
  $seenArgs = @(
    "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy",
    "--adapter-path", $FinalEvalCheckpointPath,
    "--checkpoint-type", $FinalEvalCheckpointType,
    "--episodes", "$FullSeenEpisodes",
    "--task-id-base", "0",
    "--run-name", $seenRunName,
    "--max-turns", "30",
    "--max-seq-len", "$BestMaxSeqLen",
    "--split", "eval_in_distribution"
  )
  Invoke-DetachedModal -RunName $seenRunName -Kind "sft-final-eval" -ModalArgs $seenArgs

  $unseenRunName = "sfteval_${RunTag}_unseen${FullUnseenEpisodes}"
  $unseenArgs = @(
    "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy",
    "--adapter-path", $FinalEvalCheckpointPath,
    "--checkpoint-type", $FinalEvalCheckpointType,
    "--episodes", "$FullUnseenEpisodes",
    "--task-id-base", "0",
    "--run-name", $unseenRunName,
    "--max-turns", "30",
    "--max-seq-len", "$BestMaxSeqLen",
    "--split", "eval_out_of_distribution"
  )
  Invoke-DetachedModal -RunName $unseenRunName -Kind "sft-final-eval" -ModalArgs $unseenArgs
}

function Launch-GrpoGrid {
  if (-not $BestSftCheckpoint) {
    throw "Set -BestSftCheckpoint to the chosen final 3-epoch SFT checkpoint."
  }
  modal run infra/app_alfworld_install.py --action prepare_subset_data_dir | Out-Null
  $env:CS224R_SKIP_OPENAI_SECRET = "1"
  $args = @(
    "maxrodriguez/tools/launch_grpo_grid.py",
    "--sft-adapter", $BestSftCheckpoint,
    "--run-tag", $RunTag
  )
  if ($SignedAttentionTransformerCheckpoint) {
    $args += @("--signed-attention-transformer-ckpt", $SignedAttentionTransformerCheckpoint)
  } else {
    $args += @("--include-methods", "trajectory_only,progress_delta,admissible_margin")
  }
  if ($GrpoMaxSpecs -gt 0) {
    $args += @("--max-specs", "$GrpoMaxSpecs")
  }
  $args += @("--eval-episodes", "$GrpoGridEvalEpisodes")
  python @args
}

function Launch-SignedAttentionGrid {
  $learningRates = @("0.00005", "0.0001", "0.0002")
  $hiddenSizes = @(64, 128)
  $numLayers = @(1, 2)
  $numHeads = 4
  $dropout = "0.0"
  $trainTrajectories = 256
  $valTrajectories = 64
  $maxTurns = 30

  foreach ($lr in $learningRates) {
    foreach ($hiddenSize in $hiddenSizes) {
      foreach ($nLayers in $numLayers) {
        $runName = "satf_${RunTag}_e1_lr$($lr.Replace('.', 'p'))_h${hiddenSize}_L${nLayers}_H${numHeads}_tr${trainTrajectories}_va${valTrajectories}_t${maxTurns}"
        $outputDir = "/vol/checkpoints/maxrodriguez_milestone/signed_attention/grid/$runName"
        $modalArgs = @(
          "run", "--detach", "maxrodriguez/grpo/app_signed_attention_transformer.py::train_signed_attention_transformer_model",
          "--epochs", "1",
          "--learning-rate", $lr,
          "--hidden-size", "$hiddenSize",
          "--n-layers", "$nLayers",
          "--n-heads", "$numHeads",
          "--dropout", $dropout,
          "--train-trajectories", "$trainTrajectories",
          "--val-trajectories", "$valTrajectories",
          "--max-turns", "$maxTurns",
          "--seed", "42",
          "--run-name", $runName,
          "--output-dir", $outputDir,
          "--base-model-path", $BaseModelPath
        )
        Invoke-DetachedModal -RunName $runName -Kind "signed-attention-grid" -ModalArgs $modalArgs
      }
    }
  }
}

function Launch-FinalSignedAttention {
  if ($BestSignedAttentionLearningRate -le 0.0) {
    throw "Set -BestSignedAttentionLearningRate from the winning signed-attention transformer grid run."
  }
  $runName = "satf_${RunTag}_final_e3_lr$($BestSignedAttentionLearningRate.ToString().Replace('.', 'p'))_h${BestSignedAttentionHiddenSize}_L${BestSignedAttentionNumLayers}_H${BestSignedAttentionNumHeads}_tr${BestSignedAttentionTrainTrajectories}_va${BestSignedAttentionValTrajectories}_t${BestSignedAttentionMaxTurns}"
  $outputDir = "/vol/checkpoints/maxrodriguez_milestone/signed_attention/final/$runName"
  $modalArgs = @(
    "run", "--detach", "maxrodriguez/grpo/app_signed_attention_transformer.py::train_signed_attention_transformer_model",
    "--epochs", "3",
    "--learning-rate", "$BestSignedAttentionLearningRate",
    "--hidden-size", "$BestSignedAttentionHiddenSize",
    "--n-layers", "$BestSignedAttentionNumLayers",
    "--n-heads", "$BestSignedAttentionNumHeads",
    "--dropout", "$BestSignedAttentionDropout",
    "--train-trajectories", "$BestSignedAttentionTrainTrajectories",
    "--val-trajectories", "$BestSignedAttentionValTrajectories",
    "--max-turns", "$BestSignedAttentionMaxTurns",
    "--seed", "42",
    "--run-name", $runName,
    "--output-dir", $outputDir,
    "--base-model-path", $BaseModelPath
  )
  Invoke-DetachedModal -RunName $runName -Kind "signed-attention-final" -ModalArgs $modalArgs
  Write-Host "Final signed-attention transformer checkpoint path: $outputDir"
}

function Launch-GrpoGridCore {
  if (-not $BestSftCheckpoint) {
    throw "Set -BestSftCheckpoint to the chosen final 3-epoch SFT checkpoint."
  }
  modal run infra/app_alfworld_install.py --action prepare_subset_data_dir | Out-Null
  $env:CS224R_SKIP_OPENAI_SECRET = "1"
  $args = @(
    "maxrodriguez/tools/launch_grpo_grid.py",
    "--sft-adapter", $BestSftCheckpoint,
    "--run-tag", $RunTag,
    "--include-methods", "trajectory_only,progress_delta,admissible_margin"
  )
  if ($GrpoMaxSpecs -gt 0) {
    $args += @("--max-specs", "$GrpoMaxSpecs")
  }
  $args += @("--eval-episodes", "$GrpoGridEvalEpisodes")
  python @args
}

function Launch-GrpoGridSignedAttention {
  if (-not $BestSftCheckpoint) {
    throw "Set -BestSftCheckpoint to the chosen final 3-epoch SFT checkpoint."
  }
  if (-not $SignedAttentionTransformerCheckpoint) {
    throw "Set -SignedAttentionTransformerCheckpoint to the final trained signed-attention transformer checkpoint."
  }
  modal run infra/app_alfworld_install.py --action prepare_subset_data_dir | Out-Null
  $env:CS224R_SKIP_OPENAI_SECRET = "1"
  $args = @(
    "maxrodriguez/tools/launch_grpo_grid.py",
    "--sft-adapter", $BestSftCheckpoint,
    "--run-tag", $RunTag,
    "--include-methods", "signed_attention",
    "--signed-attention-transformer-ckpt", $SignedAttentionTransformerCheckpoint
  )
  if ($GrpoMaxSpecs -gt 0) {
    $args += @("--max-specs", "$GrpoMaxSpecs")
  }
  $args += @("--eval-episodes", "$GrpoGridEvalEpisodes")
  python @args
}

function Launch-FinalGrpo {
  if (-not $BestSftCheckpoint) {
    throw "Set -BestSftCheckpoint to the chosen final 3-epoch SFT checkpoint."
  }
  if (-not $FinalGrpoMethod) {
    throw "Set -FinalGrpoMethod to the winning GRPO method."
  }
  if ($FinalGrpoAlpha -lt 0.0) {
    throw "Set -FinalGrpoAlpha to the winning GRPO alpha."
  }
  if ($FinalGrpoLearningRate -le 0.0) {
    throw "Set -FinalGrpoLearningRate to the winning GRPO learning rate."
  }
  if ($FinalGrpoKlCoeff -le 0.0) {
    throw "Set -FinalGrpoKlCoeff to the winning GRPO KL coefficient."
  }

  $runName = "grpofinal_${RunTag}_${FinalGrpoMethod}_a$($FinalGrpoAlpha.ToString().Replace('.', 'p'))_lr$($FinalGrpoLearningRate.ToString().Replace('.', 'p'))_kl$($FinalGrpoKlCoeff.ToString().Replace('.', 'p'))"
  $modalArgs = @(
    "run", "--detach", "maxrodriguez/grpo/app_alfworld_grpo.py::main",
    "--action", "launch_manual",
    "--sft-adapter", $BestSftCheckpoint,
    "--signed-attention-transformer-ckpt", $SignedAttentionTransformerCheckpoint,
    "--alpha", "$FinalGrpoAlpha",
    "--turn-reward-method", $FinalGrpoMethod,
    "--learning-rate", "$FinalGrpoLearningRate",
    "--kl-coeff", "$FinalGrpoKlCoeff",
    "--n-episodes", "$FinalGrpoEpisodes",
    "--k", "$FinalGrpoK",
    "--max-turns", "$FinalGrpoMaxTurns",
    "--clip-eps", "$FinalGrpoClipEps",
    "--grad-accum-steps", "$FinalGrpoGradAccumSteps",
    "--max-tokens-per-microbatch", "$FinalGrpoMaxTokensPerMicrobatch",
    "--kl-warmup-episodes", "$FinalGrpoKlWarmupEpisodes",
    "--task-id-stride", "$FinalGrpoTaskIdStride",
    "--dataset-size-mode", $FinalGrpoDatasetSizeMode,
    "--eval-episodes", "$FinalGrpoEvalEpisodes",
    "--run-name-suffix", "final"
  )
  Invoke-DetachedModal -RunName $runName -Kind "grpo-final" -ModalArgs $modalArgs
}

function Launch-FinalGrpoEval {
  if (-not $FinalEvalCheckpointPath) {
    throw "Set -FinalEvalCheckpointPath to the final GRPO adapter checkpoint to benchmark."
  }

  $seenRunName = "grpobench_${RunTag}_seen${FullSeenEpisodes}"
  $seenArgs = @(
    "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy",
    "--adapter-path", $FinalEvalCheckpointPath,
    "--checkpoint-type", "lora",
    "--episodes", "$FullSeenEpisodes",
    "--task-id-base", "0",
    "--run-name", $seenRunName,
    "--max-turns", "$FinalGrpoMaxTurns",
    "--max-seq-len", "$BestMaxSeqLen",
    "--split", "eval_in_distribution"
  )
  Invoke-DetachedModal -RunName $seenRunName -Kind "grpo-final-eval" -ModalArgs $seenArgs

  $unseenRunName = "grpobench_${RunTag}_unseen${FullUnseenEpisodes}"
  $unseenArgs = @(
    "run", "--detach", "maxrodriguez/supervised_FT/app_alfworld_sft_plus.py::evaluate_freeform_greedy",
    "--adapter-path", $FinalEvalCheckpointPath,
    "--checkpoint-type", "lora",
    "--episodes", "$FullUnseenEpisodes",
    "--task-id-base", "0",
    "--run-name", $unseenRunName,
    "--max-turns", "$FinalGrpoMaxTurns",
    "--max-seq-len", "$BestMaxSeqLen",
    "--split", "eval_out_of_distribution"
  )
  Invoke-DetachedModal -RunName $unseenRunName -Kind "grpo-final-eval" -ModalArgs $unseenArgs
}

function Collect-Results {
  New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null
  modal volume get cs224r-hgpo-vol /manifests/maxrodriguez $DownloadDir
  python maxrodriguez/tools/compile_milestone_results.py `
    --input-root $DownloadDir `
    --output-dir maxrodriguez/results/compiled
}

if ($Stage -in @("sft-grid", "all")) {
  Launch-SftGrid
}

if ($Stage -eq "sft-final") {
  Launch-FinalSft -UseDagger $false
}

if ($Stage -eq "sft-final-eval") {
  Launch-FinalSftEval
}

if ($Stage -eq "sft-dagger-comparison") {
  Launch-FinalSft -UseDagger $false
  Launch-FinalSft -UseDagger $true
}

if ($Stage -eq "signed-attention-grid") {
  Launch-SignedAttentionGrid
}

if ($Stage -eq "signed-attention-final") {
  Launch-FinalSignedAttention
}

if ($Stage -eq "grpo-grid") {
  Launch-GrpoGrid
}

if ($Stage -eq "grpo-grid-core") {
  Launch-GrpoGridCore
}

if ($Stage -eq "grpo-grid-signed-attention") {
  Launch-GrpoGridSignedAttention
}

if ($Stage -eq "grpo-final") {
  Launch-FinalGrpo
}

if ($Stage -eq "grpo-final-eval") {
  Launch-FinalGrpoEval
}

if ($Stage -eq "collect") {
  Collect-Results
}

if ($Stage -eq "all") {
  Write-Host ""
  Write-Host "Launched the 1-epoch SFT selection grid detached."
  Write-Host "After those jobs finish, run:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage collect"
  Write-Host "Pick the best SFT hyperparameters, then launch final SFT:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage sft-final -BestLearningRate <lr> -BestMaxSeqLen <len> -BestGradAccum <ga>"
  Write-Host "Then benchmark that final SFT on the FULL 140 seen + 134 unseen:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage sft-final-eval -FinalEvalCheckpointPath <checkpoint_path> -FinalEvalCheckpointType full -BestMaxSeqLen <len>"
  Write-Host "Then launch the signed-attention transformer search:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage signed-attention-grid"
  Write-Host "After picking the best transformer hyperparameters, run the final transformer fit:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage signed-attention-final -BestSignedAttentionLearningRate <lr> -BestSignedAttentionHiddenSize <h> -BestSignedAttentionNumLayers <L>"
  Write-Host "Then launch the core GRPO sweep from the final SFT checkpoint:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage grpo-grid-core -BestSftCheckpoint <checkpoint_path>"
  Write-Host "And the signed-attention GRPO sweep with the trained transformer:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage grpo-grid-signed-attention -BestSftCheckpoint <checkpoint_path> -SignedAttentionTransformerCheckpoint <transformer_ckpt>"
  Write-Host "After picking the best GRPO variant, run the longer final GRPO training:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage grpo-final -BestSftCheckpoint <checkpoint_path> -SignedAttentionTransformerCheckpoint <transformer_ckpt_if_needed> -FinalGrpoMethod <method> -FinalGrpoAlpha <alpha> -FinalGrpoLearningRate <lr> -FinalGrpoKlCoeff <kl>"
  Write-Host "Then benchmark that final GRPO adapter on the FULL 140 seen + 134 unseen:"
  Write-Host "  .\maxrodriguez\scripts\run_milestone_gridsearch.ps1 -Stage grpo-final-eval -FinalEvalCheckpointPath <grpo_adapter_path> -BestMaxSeqLen <len> -FinalGrpoMaxTurns <turns>"
  Write-Host ""
  Write-Host "Detached submission logs:"
  Write-Host "  $submitDir"
}
