"""H-GRPO trainer config loader (Day 14).

`build_trainer_from_config(cfg, *, policy)` builds the full Method-A/B/C
machinery from a `configs/method_hgpo_*.json`-shaped dict and returns
the assembled `HGPOTrainer` plus the producer plumbing
`infra/app_train_loop.py` needs to wire the rollout collector for
Method B.

This is a thin loader — all heavy lifting (decomposer construction,
TurnRD model init, embedder factory, judge backend wiring) is delegated
to the factories that already exist in
`src.algorithms.hgpo.decomposers.{__init__, base, judge, turnrd}`,
`src.judge.backend::build_judge`, `src.turnrd.{model, embedders}`.

torch is imported at module top because `TurnRD` instantiation requires
it. Mac-side tests gate via `pytest.importorskip("torch")`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch

from src.algorithms.grpo.kl import AdaptiveKLConfig
from src.algorithms.grpo.trainer import (
    HGPOTrainer,
    HGPOTrainerConfig,
    PerTurnDecomposer,
    progress_decomposer,
)

if TYPE_CHECKING:
    from src.algorithms.hgpo.decomposers.base import TurnRewardDecomposer
    from src.algorithms.hgpo.decomposers.turnrd import TurnEmbedder, TurnRDDecomposer
    from src.policy.lora_policy import LoRAPolicy

logger = logging.getLogger(__name__)


# Return tuple alias: keep the call-site readable.
BuildResult = Tuple[
    HGPOTrainer,
    Optional[Callable[[], None]],   # refresh_decomposer_fn
    Optional[str],                  # turnrd_emit_path
    Optional["TurnEmbedder"],       # turnrd_embedder for the producer
    Optional["TurnRewardDecomposer"],  # judge_decomposer for Mode-2 producer
]


def _build_kl_cfg(cfg: dict[str, Any]) -> AdaptiveKLConfig:
    """Map `cfg["train"]["kl_coeff"]` (existing JSON key) → `AdaptiveKLConfig`.

    The existing method configs only specify a single `kl_coeff` scalar
    that we feed into both `init_coef` and `target_kl` (the standard
    PPO-RLHF-default symmetric setup). All other AdaptiveKL knobs keep
    their dataclass defaults.
    """
    train_cfg = cfg.get("train", {}) or {}
    kl_coeff = train_cfg.get("kl_coeff")
    if kl_coeff is None:
        return AdaptiveKLConfig()
    return AdaptiveKLConfig(init_coef=float(kl_coeff), target_kl=float(kl_coeff))


def _build_progress_branch(
    trainer_cfg: HGPOTrainerConfig,
    policy: "LoRAPolicy",
) -> BuildResult:
    """Method C: progress decomposer, no refresh, no producer."""
    trainer = HGPOTrainer(
        policy=policy, decomposer=progress_decomposer, cfg=trainer_cfg
    )
    return trainer, None, None, None, None


def _build_judge_branch(
    cfg: dict[str, Any],
    trainer_cfg: HGPOTrainerConfig,
    policy: "LoRAPolicy",
) -> BuildResult:
    """Method A: build JudgeBackend + JudgeCache + JudgeDecomposer."""
    judge_cfg = cfg.get("judge")
    if not isinstance(judge_cfg, dict):
        raise ValueError(
            "build_trainer_from_config(decomposer='judge'): missing "
            "top-level 'judge' config block."
        )
    cache_cfg = judge_cfg.get("cache")
    if not isinstance(cache_cfg, dict) or "path" not in cache_cfg:
        raise ValueError(
            "build_trainer_from_config(decomposer='judge'): missing "
            "judge.cache.path."
        )

    from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer
    from src.judge.backend import build_judge
    from src.judge.cache import JudgeCache

    backend = build_judge(cfg)
    cache = JudgeCache(str(cache_cfg["path"]))
    limits_cfg = judge_cfg.get("limits", {}) if isinstance(judge_cfg, dict) else {}
    max_calls = int(limits_cfg.get("max_judge_calls_per_run", 0)) if isinstance(limits_cfg, dict) else 0
    decomposer = JudgeDecomposer(
        backend=backend,
        cache=cache,
        max_judge_calls_per_run=(max_calls or None),
    )
    trainer = HGPOTrainer(
        policy=policy,
        decomposer=decomposer.decompose,  # type: ignore[arg-type]
        cfg=trainer_cfg,
    )
    return trainer, None, None, None, None


def _build_turnrd_branch(
    cfg: dict[str, Any],
    trainer_cfg: HGPOTrainerConfig,
    policy: "LoRAPolicy",
) -> BuildResult:
    """Method B: build TurnRD model + adapter + (optional) refresh fn +
    producer plumbing.

    Reads cfg["turnrd"]:
      mode: 1 | 2
      layers: int
      hidden_size: int
      refresh_every_episodes: int
      replay_buffer_path: str | None  (forwarded as turnrd_emit_path)
      ckpt_path: str | None           (refresh fn loads this when not None)
      max_turns: int (optional; defaults to 64 — matches `TurnRDConfig`)
    """
    turnrd_cfg = cfg.get("turnrd")
    if not isinstance(turnrd_cfg, dict):
        raise ValueError(
            "build_trainer_from_config(decomposer='turnrd'): missing "
            "top-level 'turnrd' config block."
        )

    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.embedders import policy_hidden_state_embedder
    from src.turnrd.model import TurnRD, TurnRDConfig

    # Read the policy's hidden size at runtime (1536 for Qwen2.5-1.5B).
    # We DO NOT trust a JSON-supplied input_dim — the policy is the
    # source of truth.
    try:
        input_dim = int(policy.model.config.hidden_size)
    except AttributeError as e:
        raise ValueError(
            "build_trainer_from_config(decomposer='turnrd'): "
            "policy.model.config.hidden_size is not available; "
            "TurnRD needs it to size input_proj."
        ) from e

    turnrd_model = TurnRD(
        TurnRDConfig(
            n_layers=int(turnrd_cfg.get("layers", TurnRDConfig().n_layers)),
            hidden_size=int(turnrd_cfg.get("hidden_size", TurnRDConfig().hidden_size)),
            n_heads=int(turnrd_cfg.get("n_heads", TurnRDConfig().n_heads)),
            max_turns=int(turnrd_cfg.get("max_turns", TurnRDConfig().max_turns)),
            dropout=float(turnrd_cfg.get("dropout", TurnRDConfig().dropout)),
        ),
        input_dim=input_dim,
    )
    embedder = policy_hidden_state_embedder(policy)
    decomposer = TurnRDDecomposer(model=turnrd_model, embedder=embedder)

    # Refresh fn (None when no ckpt_path is configured — the trainer
    # then disables the hook even if cfg.refresh_every_episodes > 0,
    # via the constructor null check).
    ckpt_path = turnrd_cfg.get("ckpt_path")
    refresh_fn: Callable[[], None] | None = None
    if ckpt_path:
        ckpt_path_resolved = Path(ckpt_path)

        def _refresh() -> None:
            if not ckpt_path_resolved.is_file():
                logger.warning(
                    "TurnRD refresh: ckpt %s not found yet; skipping load.",
                    ckpt_path_resolved,
                )
                return
            sd = torch.load(
                ckpt_path_resolved,
                map_location=next(decomposer.model.parameters()).device,
                weights_only=True,
            )
            decomposer.load_state_dict(sd)
            logger.info("TurnRD refresh: loaded %s", ckpt_path_resolved)

        refresh_fn = _refresh

    # Producer plumbing for `infra/app_train_loop.py`.
    turnrd_emit_path = turnrd_cfg.get("replay_buffer_path")
    mode = int(turnrd_cfg.get("mode", 1))
    if mode not in (1, 2):
        raise ValueError(
            f"build_trainer_from_config: turnrd.mode must be 1 or 2; got {mode}."
        )

    judge_decomposer: "TurnRewardDecomposer | None" = None
    if mode == 2:
        # Reuse the Method-A factory so the qualified-task-id math + cache
        # read-through stays in one place. Requires a 'judge' block in the
        # config (a Method-B Mode-2 run still needs the judge-cache read
        # path to source labels).
        judge_cfg = cfg.get("judge")
        if not isinstance(judge_cfg, dict):
            raise ValueError(
                "build_trainer_from_config: turnrd.mode=2 requires a "
                "top-level 'judge' config block (the producer reads "
                "labels via JudgeDecomposer)."
            )
        cache_cfg = judge_cfg.get("cache")
        if not isinstance(cache_cfg, dict) or "path" not in cache_cfg:
            raise ValueError(
                "build_trainer_from_config: turnrd.mode=2 requires "
                "judge.cache.path."
            )
        from src.algorithms.hgpo.decomposers.judge import JudgeDecomposer
        from src.judge.backend import build_judge
        from src.judge.cache import JudgeCache

        judge_backend = build_judge(cfg)
        judge_cache = JudgeCache(str(cache_cfg["path"]))
        limits_cfg = judge_cfg.get("limits", {}) if isinstance(judge_cfg, dict) else {}
        max_calls = int(limits_cfg.get("max_judge_calls_per_run", 0)) if isinstance(limits_cfg, dict) else 0
        judge_decomposer = JudgeDecomposer(
            backend=judge_backend,
            cache=judge_cache,
            max_judge_calls_per_run=(max_calls or None),
        )

    trainer = HGPOTrainer(
        policy=policy,
        decomposer=decomposer,  # the object; __call__ fwd to .decompose
        cfg=trainer_cfg,
        refresh_decomposer_fn=refresh_fn,
    )
    return (
        trainer,
        refresh_fn,
        str(turnrd_emit_path) if turnrd_emit_path else None,
        embedder,
        judge_decomposer,
    )


def build_trainer_from_config(
    cfg: dict[str, Any],
    *,
    policy: "LoRAPolicy",
) -> BuildResult:
    """Build the H-GRPO trainer + producer plumbing from a method config.

    Returns `(trainer, refresh_fn, turnrd_emit_path, turnrd_embedder,
    judge_decomposer)`. For Methods A/C the last four are `None`.

    Reads `cfg["hgpo"]["decomposer"]`:
    - "progress": `progress_decomposer`, no refresh fn, no producer.
    - "judge":    `JudgeDecomposer(backend=build_judge(cfg), ...)`.
                  Requires `cfg["judge"]["cache"]["path"]`.
    - "turnrd":   `TurnRDDecomposer(model=TurnRD(...), embedder=
                  policy_hidden_state_embedder(policy))`. Builds a
                  refresh fn from `cfg["turnrd"]["ckpt_path"]` when set;
                  exposes producer plumbing via the last 3 returns.
                  Requires `cfg["turnrd"]` block; for Mode 2,
                  additionally requires the `cfg["judge"]` block.
    """
    train_cfg = cfg.get("train", {}) or {}
    hgpo_cfg = cfg.get("hgpo", {}) or {}
    turnrd_cfg = cfg.get("turnrd", {}) or {}

    # Build the trainer config from the existing JSON keys.
    trainer_cfg = HGPOTrainerConfig(
        alpha=float(hgpo_cfg.get("alpha", HGPOTrainerConfig().alpha)),
        lambda_consistency=float(
            hgpo_cfg.get("lambda_consistency", HGPOTrainerConfig().lambda_consistency)
        ),
        clip_eps=float(train_cfg.get("clip_eps", HGPOTrainerConfig().clip_eps)),
        learning_rate=float(
            train_cfg.get("learning_rate", HGPOTrainerConfig().learning_rate)
        ),
        grad_accum_steps=int(
            train_cfg.get("grad_accum_steps", HGPOTrainerConfig().grad_accum_steps)
        ),
        max_grad_norm=float(
            train_cfg.get("max_grad_norm", HGPOTrainerConfig().max_grad_norm)
        ),
        kl_cfg=_build_kl_cfg(cfg),
        kl_warmup_episodes=int(
            train_cfg.get("kl_warmup_episodes", HGPOTrainerConfig().kl_warmup_episodes)
        ),
        refresh_every_episodes=int(
            turnrd_cfg.get("refresh_every_episodes", HGPOTrainerConfig().refresh_every_episodes)
        ),
        turnrd_lr=float(
            turnrd_cfg.get("turnrd_lr", HGPOTrainerConfig().turnrd_lr)
        ),
    )

    name = str(hgpo_cfg.get("decomposer", "progress")).lower()
    if name == "progress":
        return _build_progress_branch(trainer_cfg, policy)
    if name == "judge":
        return _build_judge_branch(cfg, trainer_cfg, policy)
    if name == "turnrd":
        return _build_turnrd_branch(cfg, trainer_cfg, policy)
    raise ValueError(
        f"build_trainer_from_config: unknown hgpo.decomposer "
        f"{name!r}; expected 'progress' | 'judge' | 'turnrd'."
    )
