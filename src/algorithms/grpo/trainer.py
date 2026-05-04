"""HGPOTrainer: PPO-clipped policy update over K-trajectory groups, with the
H-GRPO advantage construction (proposal §3.1) and an adaptive KL controller
to a frozen reference model.

torch / transformers imports are deferred to method bodies so the module
loads cleanly on a Mac (where the heavy stack is not installed). All
gradient math runs on Modal A100 inside `webshop_image` / `image`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.algorithms.grpo.advantage import (
    combine,
    compute_traj_advantages,
    compute_turn_advantages,
    consistency_loss,
)
from src.algorithms.grpo.kl import AdaptiveKLConfig, AdaptiveKLController
from src.algorithms.grpo.rollout import TrajectoryGroup

if TYPE_CHECKING:
    import torch
    from src.policy.lora_policy import LoRAPolicy


@dataclass
class HGPOTrainerConfig:
    """Hyperparameters for one `train_step` call."""
    # H-GRPO advantage knobs (proposal §3.1)
    alpha: float = 0.5
    lambda_consistency: float = 0.1
    # PPO knobs
    clip_eps: float = 0.2
    # Optimizer
    learning_rate: float = 1e-6
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    # KL-to-reference
    kl_cfg: AdaptiveKLConfig = field(default_factory=AdaptiveKLConfig)
    # When True, recompute new-policy logprobs in this trainer; when False the
    # caller passes them in (useful for unit tests with mocked policies).
    recompute_logprobs: bool = True


@dataclass
class TrainStepStats:
    policy_loss: float = 0.0
    kl_term: float = 0.0
    consistency: float = 0.0
    total_loss: float = 0.0
    observed_kl: float = 0.0
    kl_coef: float = 0.0
    grad_norm: float = 0.0
    n_action_tokens: int = 0
    mean_traj_adv: float = 0.0
    mean_turn_adv: float = 0.0


PerTurnDecomposer = Callable[[TrajectoryGroup], list[list[float]]]
"""A turn decomposer: takes a TrajectoryGroup, returns list[K] of list[T_i]
of per-turn rewards `r̂_t^i` (Methods A/B/C). For Method C (Progress) the
decomposer just reads `traj.turns[t].raw_env_reward`.
"""


def progress_decomposer(group: TrajectoryGroup) -> list[list[float]]:
    """Method C reference: per-turn reward = raw env reward (already populated
    by the rollout collector for envs that emit progress signals)."""
    return [[float(turn.raw_env_reward) for turn in traj.turns] for traj in group.trajectories]


class HGPOTrainer:
    """Group-relative policy optimization with hierarchical (trajectory + turn)
    advantages, PPO-clip surrogate, and an adaptive KL penalty against the
    frozen base.
    """

    def __init__(
        self,
        policy: "LoRAPolicy",
        decomposer: PerTurnDecomposer,
        cfg: HGPOTrainerConfig | None = None,
    ) -> None:
        self.policy = policy
        self.decomposer = decomposer
        self.cfg = cfg or HGPOTrainerConfig()
        self.kl_controller = AdaptiveKLController(self.cfg.kl_cfg)
        self._optimizer: Any = None  # lazy-init in train_step
        self._step: int = 0

    # -----------------------------------------------------------------
    # Pure-Python advantage stage (works without torch; used by tests)
    # -----------------------------------------------------------------

    def build_advantages(self, group: TrajectoryGroup) -> dict[str, Any]:
        """Run the H-GRPO advantage construction over a TrajectoryGroup.

        Returns a dict with `traj_adv` (list[K]), `turn_adv` (list[K] of
        list[T_i]), `combined` (list[K] of list[T_i]), `consistency` (scalar).
        """
        per_turn_rewards = self.decomposer(group)
        traj_adv = compute_traj_advantages(group.final_rewards())
        turn_adv = compute_turn_advantages(per_turn_rewards)
        combined = combine(self.cfg.alpha, traj_adv, turn_adv)
        cons = consistency_loss(self.cfg.lambda_consistency, traj_adv, turn_adv)
        return {
            "traj_adv": traj_adv,
            "turn_adv": turn_adv,
            "combined": combined,
            "consistency": cons,
            "per_turn_rewards": per_turn_rewards,
        }

    # -----------------------------------------------------------------
    # Torch path (Modal A100 only; trainer.train_step)
    # -----------------------------------------------------------------

    def _ensure_optimizer(self) -> None:
        if self._optimizer is not None:
            return
        import torch  # type: ignore[import-not-found]
        # Snapshot trainable params at init time so optimizer / grad-clip /
        # AdamW state cannot drift if `trainable_parameters()` ever returns
        # a different ordering or set on a later call (e.g. after
        # `merge_and_unload()` for a vLLM weight sync). Review item M7.
        self._trainable_params: list = list(self.policy.trainable_parameters())
        self._optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=self.cfg.learning_rate,
        )

    def _new_logprobs_for_turn(
        self, prompt_token_ids: list[int], action_token_ids: list[int]
    ) -> "torch.Tensor":
        """Recompute per-token logprobs under the *current* policy for a single
        (prompt, action) pair. Returns a 1-D tensor of length len(action_token_ids)."""
        import torch  # type: ignore[import-not-found]
        from torch.nn import functional as F  # type: ignore[import-not-found]

        full = prompt_token_ids + list(action_token_ids)
        if not action_token_ids:
            return torch.zeros(0)

        device = next(self.policy.model.parameters()).device
        input_ids = torch.tensor([full], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.set_grad_enabled(True):
            logits = self.policy.model(input_ids, attention_mask=attention_mask).logits  # [1, L, V]
        # Predict action tokens from positions [len(prompt) - 1 ... L - 2]
        start = len(prompt_token_ids) - 1
        end = start + len(action_token_ids)
        # Cast to fp32 BEFORE log_softmax — bf16 has ~3 mantissa decimals and
        # the resulting logprob noise is amplified by exp() when computing
        # the importance ratio (review M3).
        slice_logits = logits[0, start:end, :].to(torch.float32)
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = torch.tensor(action_token_ids, dtype=torch.long, device=device)
        return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [Tk]

    def _ref_logprobs_for_turn(
        self, prompt_token_ids: list[int], action_token_ids: list[int]
    ) -> "torch.Tensor":
        """Per-token logprobs under the *frozen reference* policy (LoRA disabled).

        Used for the KL-to-reference penalty (review C2). PEFT's
        `disable_adapter()` context manager temporarily bypasses the LoRA
        adapter so the forward sees only the original SFT base weights.
        Returned tensor is detached (no grad needed for the reference side).
        """
        import torch  # type: ignore[import-not-found]
        from torch.nn import functional as F  # type: ignore[import-not-found]

        full = prompt_token_ids + list(action_token_ids)
        if not action_token_ids:
            return torch.zeros(0)

        device = next(self.policy.model.parameters()).device
        input_ids = torch.tensor([full], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad(), self.policy.model.disable_adapter():
            logits = self.policy.model(input_ids, attention_mask=attention_mask).logits  # [1, L, V]
        start = len(prompt_token_ids) - 1
        end = start + len(action_token_ids)
        # fp32 cast (review M3) — see _new_logprobs_for_turn for rationale.
        slice_logits = logits[0, start:end, :].to(torch.float32)
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = torch.tensor(action_token_ids, dtype=torch.long, device=device)
        return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).detach()

    def compute_loss(self, group: TrajectoryGroup) -> tuple["torch.Tensor", TrainStepStats]:
        """Compute the H-GRPO total loss for one group.

        Reads `prompt_token_ids` and `action_token_ids` directly from each
        `TurnRecord` (the rollout collector populates both as of Day 5.5).
        """
        import torch  # type: ignore[import-not-found]

        adv = self.build_advantages(group)
        combined: list[list[float]] = adv["combined"]

        device = next(self.policy.model.parameters()).device

        all_new_lp: list[torch.Tensor] = []
        all_old_lp: list[torch.Tensor] = []
        all_ref_lp: list[torch.Tensor] = []
        all_adv: list[torch.Tensor] = []
        for i, traj in enumerate(group.trajectories):
            for t, turn in enumerate(traj.turns):
                ids = list(turn.action_token_ids)
                prompt_ids = list(turn.prompt_token_ids)
                if not ids or not prompt_ids:
                    continue
                old_lp = torch.tensor(
                    turn.action_token_logprobs, dtype=torch.float32, device=device
                )
                new_lp = self._new_logprobs_for_turn(prompt_ids, ids)
                ref_lp = self._ref_logprobs_for_turn(prompt_ids, ids)
                a_t = combined[i][t]
                adv_vec = torch.full((len(ids),), float(a_t), dtype=torch.float32, device=device)
                all_new_lp.append(new_lp)
                all_old_lp.append(old_lp)
                all_ref_lp.append(ref_lp)
                all_adv.append(adv_vec)

        if not all_new_lp:
            zero = torch.zeros((), device=device, requires_grad=True)
            return zero, TrainStepStats()

        new_lp_t = torch.cat(all_new_lp).to(torch.float32)
        old_lp_t = torch.cat(all_old_lp).to(torch.float32)
        ref_lp_t = torch.cat(all_ref_lp).to(torch.float32)
        adv_t = torch.cat(all_adv).to(torch.float32)

        # PPO importance ratio uses (new vs old=rollout) — that's correct.
        ratio = torch.exp(new_lp_t - old_lp_t)
        clip_eps = self.cfg.clip_eps
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        unclipped = ratio * adv_t
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        # KL penalty uses (new vs ref=frozen base) — anchors policy to SFT prior
        # per proposal §3.1 (review item C2). k3 estimator: kl = (rho - 1) - log(rho),
        # non-negative per-token AND unbiased estimator of KL(ref||new).
        ref_log_ratio = new_lp_t - ref_lp_t
        ref_ratio = torch.exp(ref_log_ratio)
        kl_per_tok = (ref_ratio - 1.0) - ref_log_ratio
        observed_kl = float(kl_per_tok.mean().detach().item())
        kl_coef = self.kl_controller.coef
        kl_term = kl_coef * kl_per_tok.mean()

        cons = float(adv["consistency"])
        # Consistency reg has zero gradient for Methods A/C — see comment below.
        # Excluded from `total` (review C3). Will be re-added on TurnRD's params
        # when Method B lands.
        total = policy_loss + kl_term

        traj_adv = adv["traj_adv"]
        flat_turn_adv: list[float] = [v for row in adv["turn_adv"] for v in row]
        stats = TrainStepStats(
            policy_loss=float(policy_loss.detach().item()),
            kl_term=float(kl_term.detach().item()),
            consistency=cons,
            total_loss=float(total.detach().item()),
            observed_kl=observed_kl,
            kl_coef=kl_coef,
            grad_norm=0.0,
            n_action_tokens=int(new_lp_t.numel()),
            mean_traj_adv=(sum(traj_adv) / max(1, len(traj_adv))),
            mean_turn_adv=(sum(flat_turn_adv) / max(1, len(flat_turn_adv))),
        )
        return total, stats

    def train_step(self, group: TrajectoryGroup) -> TrainStepStats:
        """One AdamW step on a single TrajectoryGroup.

        Honors `cfg.grad_accum_steps` (review M4): the loss is divided by
        `grad_accum_steps` and `optimizer.step()` is only called every Nth
        invocation. `n_action_tokens == 0` short-circuits to a no-op
        (review nit) so we don't crash on a leaf-tensor backward.
        """
        import torch  # type: ignore[import-not-found]

        self._ensure_optimizer()
        self.policy.model.train()

        loss, stats = self.compute_loss(group)
        if stats.n_action_tokens == 0:
            # No live action tokens this group; nothing to learn from.
            new_coef = self.kl_controller.update(stats.observed_kl)
            stats.kl_coef = new_coef
            self._step += 1
            return stats

        accum = max(1, int(self.cfg.grad_accum_steps))
        scaled_loss = loss / accum
        scaled_loss.backward()

        # Only step the optimizer every `accum` train_step calls; otherwise
        # accumulate grads silently.
        is_step_boundary = ((self._step + 1) % accum) == 0
        if is_step_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_params, self.cfg.max_grad_norm
            )
            stats.grad_norm = float(grad_norm)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
        else:
            stats.grad_norm = 0.0  # not yet a step boundary

        new_coef = self.kl_controller.update(stats.observed_kl)
        stats.kl_coef = new_coef
        self._step += 1
        return stats
