"""H-GRPO trainer config loader.

`build_trainer_from_config` builds the trainer (and Method-B producer
plumbing) from a method config dict, delegating to the decomposer
factories.
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
    from src.algorithms.hgpo.decomposers.turnrd import TurnEmbedder
    from src.policy.lora_policy import LoRAPolicy

logger = logging.getLogger(__name__)


BuildResult = Tuple[
    HGPOTrainer,
    Optional[Callable[[], None]],   # refresh_decomposer_fn
    Optional[str],                  # turnrd_emit_path
    Optional["TurnEmbedder"],       # turnrd_embedder for the producer
    Optional["TurnRewardDecomposer"],  # judge_decomposer for Mode-2 producer
]


def _build_kl_cfg(cfg: dict[str, Any]) -> AdaptiveKLConfig:
    """Map `cfg["train"]["kl_coeff"]` into both `init_coef` and `target_kl`; other knobs keep defaults."""
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
    """Method B: build the TurnRD model + adapter, an optional ckpt refresh
    fn, and producer plumbing. Reads architecture, mode, and ckpt settings
    from cfg["turnrd"] (version "v1" or "v2").
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

    # Hidden size comes from the policy at runtime, not from JSON.
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
        # v2: bidirectional encoder + per-turn heads; reuses v1's JSON keys.
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
                # Optional FiLM goal conditioning for TurnRDv2.
                goal_conditioned_value_head=bool(
                    turnrd_cfg.get(
                        "goal_conditioned_value_head",
                        v2_defaults.goal_conditioned_value_head,
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

    # Refresh fn stays None without a ckpt_path, which disables the hook.
    ckpt_path = turnrd_cfg.get("ckpt_path")
    refresh_fn: Callable[[], None] | None = None
    # Store the most recent checkpoint-refresh status for run manifests.
    decomposer._last_refresh_status = None  # type: ignore[attr-defined]
    if ckpt_path:
        ckpt_path_resolved = Path(ckpt_path)

        def _refresh() -> None:
            # Reload the Modal Volume so this container sees a ckpt committed
            # by another container (Volumes are eventually-consistent).
            try:
                from infra.common import volume as _shared_volume  # type: ignore[import-not-found]

                _shared_volume.reload()
            except Exception:
                # `infra.common` exists only inside Modal; elsewhere reload is a no-op.
                pass
            if not ckpt_path_resolved.is_file():
                logger.warning(
                    "TurnRD refresh: ckpt %s not found yet; skipping load.",
                    ckpt_path_resolved,
                )
                decomposer._last_refresh_status = {  # type: ignore[attr-defined]
                    "loaded": False,
                    "path": str(ckpt_path_resolved),
                    "reason": "ckpt_not_found",
                }
                return
            sd = torch.load(
                ckpt_path_resolved,
                map_location=next(decomposer.model.parameters()).device,
                weights_only=True,
            )
            # strict=False tolerates FiLM schema differences; warn on key mismatches.
            _load_result = decomposer.load_state_dict(sd, strict=False)
            try:
                _missing = list(getattr(_load_result, "missing_keys", []) or [])
                _unexpected = list(getattr(_load_result, "unexpected_keys", []) or [])
                if _missing or _unexpected:
                    logger.warning(
                        "TurnRD refresh: load_state_dict(strict=False) -> "
                        "missing=%d, unexpected=%d (missing example: %s; "
                        "unexpected example: %s)",
                        len(_missing),
                        len(_unexpected),
                        _missing[:3],
                        _unexpected[:3],
                    )
            except Exception:
                pass
            try:
                _stat = ckpt_path_resolved.stat()
                _size = int(_stat.st_size)
                _mtime = float(_stat.st_mtime)
            except OSError:
                _size = -1
                _mtime = -1.0
            decomposer._last_refresh_status = {  # type: ignore[attr-defined]
                "loaded": True,
                "path": str(ckpt_path_resolved),
                "size_bytes": _size,
                "mtime": _mtime,
            }
            logger.info(
                "TurnRD refresh: loaded %s (size=%d B, mtime=%.0f)",
                ckpt_path_resolved,
                _size,
                _mtime,
            )

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
        # Reuse the Method-A factory; Mode-2 sources labels via the judge cache.
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

    # Eager startup load: the in-loop refresh hook skips step 0, so a round
    # in a fresh container would never load the between-rounds ckpt. Load
    # once here; no-op when the ckpt is absent.
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
    """Method D: counterfactual-rollout decomposer. Reuses the collector's
    runner and env factory (no second vLLM init) and reads
    cfg["counterfactual"] for rollout settings.
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
        decomposer=cf_decomposer,  # __call__ -> .decompose
        cfg=trainer_cfg,
    )
    # No TurnRD producer plumbing for Method D.
    return trainer, None, None, None, None


def build_trainer_from_config(
    cfg: dict[str, Any],
    *,
    policy: "LoRAPolicy",
    # CF-only deps; ignored by Methods A/B/C.
    runner: Optional["_RunnerLike"] = None,
    env_factory: Optional["EnvFactory"] = None,
    prompt_renderer: Optional["PromptRenderer"] = None,
    action_parser: Optional["ActionParser"] = None,
    sampling_factory: Optional["SamplingFactory"] = None,
) -> BuildResult:
    """Build the H-GRPO trainer + producer plumbing from a method config.

    Dispatches on `cfg["hgpo"]["decomposer"]` ("progress" | "judge" |
    "turnrd" | "counterfactual"). Returns `(trainer, refresh_fn,
    turnrd_emit_path, turnrd_embedder, judge_decomposer)`; the last four
    are populated only by the turnrd branch.
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
        use_v_projection_for_decomposition=bool(
            hgpo_cfg.get(
                "use_v_projection_for_decomposition",
                HGPOTrainerConfig().use_v_projection_for_decomposition,
            )
        ),
        v_projection_clamp=float(
            hgpo_cfg.get(
                "v_projection_clamp",
                HGPOTrainerConfig().v_projection_clamp,
            )
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
        f"{name!r}; expected 'progress' | 'judge' | 'turnrd' | "
        "'counterfactual'."
    )
