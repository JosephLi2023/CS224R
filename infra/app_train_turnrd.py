"""Modal A100 app: standalone TurnRD trainer (Method B).

Wraps `src.turnrd.train.train_turnrd` so the standalone Method-B
fitting pass can run on a single A100 against a replay JSONL written
by the in-loop producer (see `src/algorithms/grpo/collectors.py`'s
emit hook).

Usage:

  modal run infra/app_train_turnrd.py \\
    --replay /vol/cache/turnrd_replay.jsonl \\
    --mode 1 \\
    --n-epochs 5 \\
    --batch-size 16 \\
    --lr 1e-4 \\
    --ckpt-out /vol/manifests/turnrd_ckpt.pt

After the checkpoint is written, the parent H-GRPO trainer's refresh
hook (built by `src.trainers.train_hgpo.build_trainer_from_config`
when `cfg["turnrd"]["ckpt_path"]` is set) reloads it on the configured
cadence.

Cost: ~$0.50 for 5 epochs over a 200-trajectory replay (A100 ~5 min).
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-train-turnrd")


@app.function(image=image, gpu="A100-80GB", volumes={VOLUME_MOUNT: volume}, timeout=30 * 60)
def train_turnrd_run(
    replay: str,
    mode: int = 1,
    n_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    ckpt_out: str = "",
    max_records: int = 0,
    # Architecture selector. "v1" = legacy TurnRD; "v2" = TurnRDv2
    # (bidirectional + identifiable R-loss + progress-prior init).
    version: str = "v1",
    # TurnRD model knobs (defaults match TurnRDConfig + the existing
    # configs/method_hgpo_turnrd.json post-improvement values).
    layers: int = 6,
    hidden_size: int = 384,
    n_heads: int = 4,
    max_turns: int = 64,
    dropout: float = 0.1,
    causal: bool = True,
    value_head: bool = True,
    # v2-only model knob (ignored when version="v1").
    progress_prior_strength: float = 1.0,
    # Aux-loss knobs (Mode 1 only). lambda_value/gamma drive the
    # per-turn V-head loss; lambda_entropy is the negative-entropy reg.
    lambda_value: float = 0.5,
    gamma: float = 0.95,
    lambda_entropy: float = 0.01,
    # v8 Tier 1: contrastive aux loss knobs.
    lambda_contrastive: float = 0.1,
    contrastive_temperature: float = 0.1,
    # v2 loss-mix knobs (effective only when version="v2").
    lambda_rank: float = 0.1,
    lambda_progress: float = 0.01,
    rank_margin: float = 0.1,
    # The producer pre-embeds turns; the standalone trainer doesn't need
    # the LoRA policy. We DO need the embedding width D, which the
    # producer wrote into the replay (we read it off the first record).
) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")

    import torch  # type: ignore[import-not-found]

    # Modal Volumes are eventually-consistent across containers. Reload
    # at startup so we see the latest replay JSONL written by the parent
    # train_loop in the previous orchestration round (which called
    # volume.commit() before exiting).
    volume.reload()

    from src.turnrd.model import TurnRD, TurnRDConfig, TurnRDv2, TurnRDv2Config
    from src.turnrd.train import train_turnrd

    if mode not in (1, 2):
        raise ValueError(f"--mode must be 1 or 2; got {mode}")
    version_norm = str(version).lower()
    if version_norm not in ("v1", "v2"):
        raise ValueError(f"--version must be 'v1' or 'v2'; got {version!r}")

    # Read the embedding width from the first record in the replay so
    # the model's input_proj is sized correctly without forcing the
    # caller to pass it on the CLI.
    with open(replay) as fh:
        first = json.loads(fh.readline())
    if not first.get("turn_embeds"):
        raise ValueError(
            f"Replay {replay} first record has no turn_embeds; cannot "
            "infer embedding width."
        )
    input_dim = len(first["turn_embeds"][0])
    print(f">>> Inferred input_dim={input_dim} from {replay}")

    torch.manual_seed(0)
    if version_norm == "v2":
        model: "TurnRD | TurnRDv2" = TurnRDv2(
            TurnRDv2Config(
                n_layers=layers,
                hidden_size=hidden_size,
                n_heads=n_heads,
                max_turns=max_turns,
                dropout=dropout,
                causal=causal,
                progress_prior_strength=progress_prior_strength,
            ),
            input_dim=input_dim,
        )
    else:
        model = TurnRD(
            TurnRDConfig(
                n_layers=layers,
                hidden_size=hidden_size,
                n_heads=n_heads,
                max_turns=max_turns,
                dropout=dropout,
                causal=causal,
                value_head=value_head,
            ),
            input_dim=input_dim,
        )
    if torch.cuda.is_available():
        model.to("cuda:0")

    t0 = time.time()
    summary = train_turnrd(
        replay,
        mode=mode,
        model=model,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        log_every=50,
        ckpt_path=(ckpt_out or None),
        max_records=(max_records or None),
        version=version_norm,
        lambda_value=lambda_value,
        gamma=gamma,
        lambda_entropy=lambda_entropy,
        lambda_contrastive=lambda_contrastive,
        contrastive_temperature=contrastive_temperature,
        lambda_rank=lambda_rank,
        lambda_progress=lambda_progress,
        rank_margin=rank_margin,
    )
    elapsed = round(time.time() - t0, 2)
    summary["elapsed_s"] = elapsed
    print(json.dumps(summary, indent=2))

    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    replay: str = "/vol/cache/turnrd_replay.jsonl",
    mode: int = 1,
    n_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    ckpt_out: str = "",
    max_records: int = 0,
    version: str = "v1",
    layers: int = 6,
    hidden_size: int = 384,
    n_heads: int = 4,
    max_turns: int = 64,
    dropout: float = 0.1,
    causal: bool = True,
    value_head: bool = True,
    progress_prior_strength: float = 1.0,
    lambda_value: float = 0.5,
    gamma: float = 0.95,
    lambda_entropy: float = 0.01,
    lambda_contrastive: float = 0.1,
    contrastive_temperature: float = 0.1,
    lambda_rank: float = 0.1,
    lambda_progress: float = 0.01,
    rank_margin: float = 0.1,
) -> None:
    import json as _json

    print(_json.dumps(
        train_turnrd_run.remote(
            replay=replay,
            mode=mode,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            ckpt_out=ckpt_out,
            max_records=max_records,
            version=version,
            layers=layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            max_turns=max_turns,
            dropout=dropout,
            causal=causal,
            value_head=value_head,
            progress_prior_strength=progress_prior_strength,
            lambda_value=lambda_value,
            gamma=gamma,
            lambda_entropy=lambda_entropy,
            lambda_contrastive=lambda_contrastive,
            contrastive_temperature=contrastive_temperature,
            lambda_rank=lambda_rank,
            lambda_progress=lambda_progress,
            rank_margin=rank_margin,
        ),
        indent=2,
        default=str,
    ))
