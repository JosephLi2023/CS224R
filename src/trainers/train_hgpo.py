"""H-GRPO trainer config loader.

`build_trainer_from_config(cfg, *, policy)` builds the full Method-A/B/C
machinery from a `configs/method_hgpo_*.json`-shaped dict and returns
the assembled `HGPOTrainer` plus the producer plumbing
`infra/app_train_loop.py` needs to wire the rollout collector for
Method B.

This is a thin loader — all heavy lifting (decomposer construction,
TurnRD model init, embedder factory, judge backend wiring) is delegated
to the factories that already exist in
`src.algorithms.hgpo.decomposers.{__init__, base, judge, turnrd,
counterfactual}`, `src.judge.backend::build_judge`,
`src.turnrd.{model, embedders}`.

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
    from src.algorithms.hgpo.decomposers.counterfactual import (
        ActionParser,
        EnvFactory,
        PromptRenderer,
        SamplingFactory,
        _RunnerLike,
    )
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
      version: "v1" | "v2" (default "v1") — selects architecture.
      mode: 1 | 2
      layers: int
      hidden_size: int
      refresh_every_episodes: int
      replay_buffer_path: str | None  (forwarded as turnrd_emit_path)
      ckpt_path: str | None           (refresh fn loads this when not None)
      max_turns: int (optional; defaults to 64 — matches `TurnRDConfig`)
      progress_prior_strength: float  (v2 only; defaults to TurnRDv2Config default)
    """
    turnrd_cfg = cfg.get("turnrd")
    if not isinstance(turnrd_cfg, dict):
        raise ValueError(
            "build_trainer_from_config(decomposer='turnrd'): missing "
            "top-level 'turnrd' config block."
        )

    from src.algorithms.hgpo.decomposers.turnrd import TurnRDDecomposer
    from src.turnrd.embedders import policy_hidden_state_embedder
    from src.turnrd.model import TurnRD, TurnRDConfig, TurnRDv2, TurnRDv2Config

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

    version = str(turnrd_cfg.get("version", "v1")).lower()
    if version not in ("v1", "v2"):
        raise ValueError(
            f"build_trainer_from_config: turnrd.version must be 'v1' or 'v2'; "
            f"got {version!r}."
        )

    if version == "v2":
        # v2 architecture: bidirectional encoder + per-turn score/value heads,
        # progress-prior init bias. Reuses the same JSON keys for the encoder
        # geometry (n_layers/hidden_size/n_heads/max_turns/dropout/causal) so
        # configs can swap version with minimal churn.
        v2_defaults = TurnRDv2Config()
        turnrd_model: "TurnRD | TurnRDv2" = TurnRDv2(
            TurnRDv2Config(
                n_layers=int(turnrd_cfg.get("layers", v2_defaults.n_layers)),
                hidden_size=int(turnrd_cfg.get("hidden_size", v2_defaults.hidden_size)),
                n_heads=int(turnrd_cfg.get("n_heads", v2_defaults.n_heads)),
                max_turns=int(turnrd_cfg.get("max_turns", v2_defaults.max_turns)),
                dropout=float(turnrd_cfg.get("dropout", v2_defaults.dropout)),
                causal=bool(turnrd_cfg.get("causal", v2_defaults.causal)),
                progress_prior_strength=float(
                    turnrd_cfg.get(
                        "progress_prior_strength", v2_defaults.progress_prior_strength
                    )
                ),
            ),
            input_dim=input_dim,
        )
    else:
        turnrd_model = TurnRD(
            TurnRDConfig(
                n_layers=int(turnrd_cfg.get("layers", TurnRDConfig().n_layers)),
                hidden_size=int(turnrd_cfg.get("hidden_size", TurnRDConfig().hidden_size)),
                n_heads=int(turnrd_cfg.get("n_heads", TurnRDConfig().n_heads)),
                max_turns=int(turnrd_cfg.get("max_turns", TurnRDConfig().max_turns)),
                dropout=float(turnrd_cfg.get("dropout", TurnRDConfig().dropout)),
                causal=bool(turnrd_cfg.get("causal", TurnRDConfig().causal)),
                value_head=bool(turnrd_cfg.get("value_head", TurnRDConfig().value_head)),
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
            # The standalone train_turnrd writes the ckpt in a separate
            # Modal container and calls volume.commit() before exiting.
            # Reload here so this container's view of /vol/cache/ picks
            # up the freshly-written ckpt (Modal Volumes are
            # eventually-consistent across containers). No-op when the
            # parent train_loop and the standalone trainer happen to
            # share a container (single-process tests).
            try:
                from infra.common import volume as _shared_volume  # type: ignore[import-not-found]

                _shared_volume.reload()
            except Exception:
                # `infra.common` is only importable inside a Modal
                # container. Outside of Modal (e.g. CPU smoke test) the
                # ckpt is on the local filesystem and reload is a no-op
                # by definition — silently skip.
                pass
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

    # Eager startup load: the trainer's in-loop refresh hook only fires
    # when `self._step % refresh_every_episodes == 0 AND self._step > 0`,
    # so for round-based protocols where each round runs in a FRESH Modal
    # container (`self._step` resets to 0), the hook would never load the
    # standalone-fitter's checkpoint written between rounds. Load it once
    # here so the very first episode of the round uses the latest fit.
    # No-op when ckpt doesn't exist yet (Round 0 of the first protocol run).
    if refresh_fn is not None:
        try:
            refresh_fn()
        except Exception as exc:  # pragma: no cover (defensive)
            logger.warning("Eager startup refresh failed: %s", exc)

    return (
        trainer,
        refresh_fn,
        str(turnrd_emit_path) if turnrd_emit_path else None,
        embedder,
        judge_decomposer,
    )


def _build_counterfactual_branch(
    cfg: dict[str, Any],
    trainer_cfg: HGPOTrainerConfig,
    policy: "LoRAPolicy",
    *,
    runner: Optional["_RunnerLike"],
    env_factory: Optional["EnvFactory"],
    prompt_renderer: Optional["PromptRenderer"],
    action_parser: Optional["ActionParser"],
    sampling_factory: Optional["SamplingFactory"],
) -> BuildResult:
    """Method D: counterfactual-rollout decomposer.

    Reuses the SAME runner the rollout collector drives so we don't pay
    a second vLLM init. The env factory must be the SAME function the
    collector uses (it's already shared in `app_train_loop.py` —
    `env_factory = lambda: WebShopAdapter(...)`); the CF decomposer
    builds its own env pool from it for the replay rollouts.

    Reads `cfg["counterfactual"]` (forwarded to
    `build_counterfactual_decomposer`):
      n_alt_actions: int       (default 2)
      max_completion_turns: int (default 3)
      cf_temperature: float    (default 1.0)
      completion_temperature: float (default 0.0)
      cf_max_tokens: int       (default 48)
      n_turns_per_traj: int    (default 0 = all turns)
      skip_if_zero_R: bool     (default True)
      output_mode: "raw_delta" | "normalized" (default "raw_delta")
      seed: int                (default 0)
    """
    if (
        runner is None
        or env_factory is None
        or prompt_renderer is None
        or action_parser is None
        or sampling_factory is None
    ):
        raise ValueError(
            "build_trainer_from_config(decomposer='counterfactual'): all of "
            "`runner`, `env_factory`, `prompt_renderer`, `action_parser`, and "
            "`sampling_factory` must be provided. The rollout collector's "
            "deps must be threaded through to the trainer builder when using "
            "the CF decomposer."
        )

    from src.algorithms.hgpo.decomposers.counterfactual import (
        build_counterfactual_decomposer,
    )

    cf_decomposer = build_counterfactual_decomposer(
        cfg,
        runner=runner,
        env_factory=env_factory,
        prompt_renderer=prompt_renderer,
        action_parser=action_parser,
        sampling_factory=sampling_factory,
    )
    trainer = HGPOTrainer(
        policy=policy,
        decomposer=cf_decomposer,  # __call__ → .decompose
        cfg=trainer_cfg,
    )
    # Method D doesn't need TurnRD producer plumbing; mirror Methods A/C.
    return trainer, None, None, None, None


def build_trainer_from_config(
    cfg: dict[str, Any],
    *,
    policy: "LoRAPolicy",
    # Optional CF-only deps — passed through from `app_train_loop.py`
    # only when `cfg["hgpo"]["decomposer"] == "counterfactual"`. Methods
    # A/B/C ignore these.
    runner: Optional["_RunnerLike"] = None,
    env_factory: Optional["EnvFactory"] = None,
    prompt_renderer: Optional["PromptRenderer"] = None,
    action_parser: Optional["ActionParser"] = None,
    sampling_factory: Optional["SamplingFactory"] = None,
) -> BuildResult:
    """Build the H-GRPO trainer + producer plumbing from a method config.

    Returns `(trainer, refresh_fn, turnrd_emit_path, turnrd_embedder,
    judge_decomposer)`. For Methods A/C/D the last four are `None`.

    Reads `cfg["hgpo"]["decomposer"]`:
    - "progress":      `progress_decomposer`, no refresh fn, no producer.
    - "judge":         `JudgeDecomposer(backend=build_judge(cfg), ...)`.
                       Requires `cfg["judge"]["cache"]["path"]`.
    - "turnrd":        `TurnRDDecomposer(model=TurnRD(...), embedder=
                       policy_hidden_state_embedder(policy))`. Builds a
                       refresh fn from `cfg["turnrd"]["ckpt_path"]` when set;
                       exposes producer plumbing via the last 3 returns.
                       Requires `cfg["turnrd"]` block; for Mode 2,
                       additionally requires the `cfg["judge"]` block.
    - "counterfactual": `CounterFactualDecomposer(runner, env_factory, ...)`.
                       Requires the caller to thread `runner` + `env_factory`
                       + `prompt_renderer` + `action_parser` +
                       `sampling_factory` through the kwargs since the CF
                       decomposer drives short alt rollouts on the same
                       runner the collector uses. See
                       `src/algorithms/hgpo/decomposers/counterfactual.py`.
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
        max_tokens_per_microbatch=int(
            train_cfg.get(
                "max_tokens_per_microbatch",
                HGPOTrainerConfig().max_tokens_per_microbatch,
            )
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
    if name == "counterfactual":
        return _build_counterfactual_branch(
            cfg,
            trainer_cfg,
            policy,
            runner=runner,
            env_factory=env_factory,
            prompt_renderer=prompt_renderer,
            action_parser=action_parser,
            sampling_factory=sampling_factory,
        )
    raise ValueError(
        f"build_trainer_from_config: unknown hgpo.decomposer "
        f"{name!r}; expected 'progress' | 'judge' | 'turnrd' | 'counterfactual'."
    )
