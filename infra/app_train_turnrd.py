"""Modal A100 app: standalone TurnRD trainer (Method B).

Wraps `src.turnrd.train.train_turnrd` to fit on a single A100 against a replay
JSONL written by the in-loop producer; the parent H-GRPO trainer reloads the
saved ckpt on its configured cadence.
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
    # Architecture: "v1" legacy TurnRD, "v2" TurnRDv2.
    version: str = "v1",
    # TurnRD model knobs (defaults match TurnRDConfig).
    layers: int = 6,
    hidden_size: int = 384,
    n_heads: int = 4,
    max_turns: int = 64,
    dropout: float = 0.1,
    causal: bool = True,
    value_head: bool = True,
    # v2-only model knob (ignored when version="v1").
    progress_prior_strength: float = 1.0,
    # Opt-in FiLM goal-conditioned V-head (v2 only); default False = legacy.
    goal_conditioned_value_head: bool = False,
    # Aux-loss knobs (Mode 1 only): per-turn V-head loss + entropy reg.
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
    # Optional replay recency decay (per-batch loss scaling); 0.0 disables.
    recency_decay_half_life: float = 0.0,
    legacy_decay_weight: float = 0.5,
    min_batch_weight: float = 1e-3,
    # Cumulative warm-start: when set + file exists, load the prior round's
    # ckpt with strict=False before training. Default "" = cold start.
    ckpt_in: str = "",
    # LR schedule; default "constant" is a no-op.
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    # Fresh-emphasis pass; 0/0 = disabled.
    fresh_emphasis_window_rounds: int = 0,
    fresh_emphasis_n_epochs: int = 0,
    # The producer pre-embeds turns; we read the embedding width off the
    # first replay record (no LoRA policy needed here).
) -> dict:
    import json
    import sys
    import time

    sys.path.insert(0, "/workspace")

    import torch  # type: ignore[import-not-found]

    # Reload so we see the replay the previous round committed.
    volume.reload()

    from src.turnrd.model import TurnRD, TurnRDConfig, TurnRDv2, TurnRDv2Config
    from src.turnrd.train import train_turnrd

    if mode not in (1, 2):
        raise ValueError(f"--mode must be 1 or 2; got {mode}")
    version_norm = str(version).lower()
    if version_norm not in ("v1", "v2"):
        raise ValueError(f"--version must be 'v1' or 'v2'; got {version!r}")

    # Read embedding width from the first replay record to size input_proj.
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
                goal_conditioned_value_head=bool(goal_conditioned_value_head),
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

    # Warm-start: load the prior round's ckpt (strict=False so a legacy
    # no-FiLM ckpt can seed a FiLM model).
    import os as _os
    if ckpt_in and _os.path.exists(ckpt_in):
        ckpt_in_state = torch.load(ckpt_in, map_location="cpu", weights_only=True)
        result = model.load_state_dict(ckpt_in_state, strict=False)
        n_loaded = len(ckpt_in_state) - len(result.missing_keys)
        print(
            f">>> warm-started from {ckpt_in} "
            f"({n_loaded}/{len(ckpt_in_state)} tensors loaded; "
            f"missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)})",
            flush=True,
        )
    elif ckpt_in:
        print(
            f">>> WARNING: ckpt_in={ckpt_in!r} does not exist; "
            "cold-starting TurnRD from random init.",
            flush=True,
        )

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
        recency_decay_half_life=(
            float(recency_decay_half_life)
            if float(recency_decay_half_life) > 0.0
            else None
        ),
        legacy_decay_weight=float(legacy_decay_weight),
        min_batch_weight=float(min_batch_weight),
        warmup_steps=int(warmup_steps),
        lr_schedule=str(lr_schedule),
        fresh_emphasis_window_rounds=int(fresh_emphasis_window_rounds),
        fresh_emphasis_n_epochs=int(fresh_emphasis_n_epochs),
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
    goal_conditioned_value_head: bool = False,
    lambda_value: float = 0.5,
    gamma: float = 0.95,
    lambda_entropy: float = 0.01,
    lambda_contrastive: float = 0.1,
    contrastive_temperature: float = 0.1,
    lambda_rank: float = 0.1,
    lambda_progress: float = 0.01,
    rank_margin: float = 0.1,
    recency_decay_half_life: float = 0.0,
    legacy_decay_weight: float = 0.5,
    min_batch_weight: float = 1e-3,
    ckpt_in: str = "",
    warmup_steps: int = 0,
    lr_schedule: str = "constant",
    fresh_emphasis_window_rounds: int = 0,
    fresh_emphasis_n_epochs: int = 0,
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
            goal_conditioned_value_head=goal_conditioned_value_head,
            lambda_value=lambda_value,
            gamma=gamma,
            lambda_entropy=lambda_entropy,
            lambda_contrastive=lambda_contrastive,
            contrastive_temperature=contrastive_temperature,
            lambda_rank=lambda_rank,
            lambda_progress=lambda_progress,
            rank_margin=rank_margin,
            recency_decay_half_life=recency_decay_half_life,
            legacy_decay_weight=legacy_decay_weight,
            min_batch_weight=min_batch_weight,
            ckpt_in=ckpt_in,
            warmup_steps=warmup_steps,
            lr_schedule=lr_schedule,
            fresh_emphasis_window_rounds=fresh_emphasis_window_rounds,
            fresh_emphasis_n_epochs=fresh_emphasis_n_epochs,
        ),
        indent=2,
        default=str,
    ))
