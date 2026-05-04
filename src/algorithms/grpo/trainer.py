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
    # Skip the KL penalty (zero it out, but still observe & log) for the
    # first N train_step calls. Useful right after SFT warm-start, when
    # the policy is intentionally far from the C2 frozen-base reference
    # and the k3 estimator can blow up to ~1e6 magnitudes. Default 0 =
    # no warmup (preserves prior behavior for tests).
    kl_warmup_episodes: int = 0
    # When True, recompute new-policy logprobs in this trainer; when False the
    # caller passes them in (useful for unit tests with mocked policies).
    recompute_logprobs: bool = True
    # Cap on total tokens (sum of len(prompt) + len(action) across rows) per
    # padded forward in `_batched_logprobs`. Keeps activation memory within
    # GPU bounds when K×T padded sequences would otherwise OOM A100-80GB
    # while sharing the GPU with vLLM's ~31 GiB KV cache.
    # Default 2048 keeps forward+backward activations under ~10 GiB for
    # Qwen2.5-1.5B; raise on larger cards or single-tenant GPUs.
    max_tokens_per_microbatch: int = 2048


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
        # Optional snapshot of LoRA weights to use as KL reference. When set,
        # `_ref_logprobs_for_turn` / batched ref forwards swap these tensors
        # in for the duration of the forward (no_grad) and restore the live
        # weights afterwards. None ⇒ ref is the C2 frozen-base path
        # (LoRA disabled). Populate via `snapshot_current_lora_as_ref()`.
        self._ref_lora_snapshot: dict[str, dict[str, Any]] | None = None

    def snapshot_current_lora_as_ref(self) -> int:
        """Capture the current LoRA tensors as the frozen KL reference.

        Returns the number of LoRA modules snapshotted. Call this AFTER
        loading an SFT-warm adapter and BEFORE the first train_step so the
        KL term penalises drift from the SFT-trained policy (not the raw
        base Qwen — which would otherwise blow up the k3 estimator).
        """
        import torch  # type: ignore[import-not-found]
        adapter_name = "default"
        snapshot: dict[str, dict[str, Any]] = {}
        with torch.no_grad():
            for name, module in self.policy.model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "scaling"):
                    snapshot[name] = {
                        "lora_A": module.lora_A[adapter_name].weight.detach().clone(),
                        "lora_B": module.lora_B[adapter_name].weight.detach().clone(),
                        "scaling": float(module.scaling[adapter_name]),
                    }
        self._ref_lora_snapshot = snapshot
        return len(snapshot)

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

    def _batched_logprobs(
        self,
        prompt_action_pairs: list[tuple[list[int], list[int]]],
        *,
        use_ref: bool,
    ) -> list["torch.Tensor"]:
        """Padded forward → per-turn action logprobs (review M6).

        Splits the K×T (prompt, action) pairs into micro-batches whose total
        token count stays under `cfg.max_tokens_per_microbatch`. Each
        micro-batch is one padded forward; activations are released between
        micro-batches so we don't OOM at K×T scale (M6 follow-up).
        """
        import torch  # type: ignore[import-not-found]
        from torch.nn import functional as F  # type: ignore[import-not-found]

        device = next(self.policy.model.parameters()).device
        pad_id = int(getattr(self.policy.tokenizer, "pad_token_id", 0) or 0)
        budget = max(1, int(self.cfg.max_tokens_per_microbatch))

        N = len(prompt_action_pairs)
        out: list[torch.Tensor] = [torch.zeros(0, device=device) for _ in range(N)]

        # Filter empty + remember original positions, sorted by length to
        # reduce padding waste within micro-batches.
        keep: list[tuple[int, list[int], list[int], int]] = []
        for idx, (prompt_ids, action_ids) in enumerate(prompt_action_pairs):
            if prompt_ids and action_ids:
                seq_len = len(prompt_ids) + len(action_ids)
                keep.append((idx, prompt_ids, action_ids, seq_len))
        if not keep:
            return out
        keep.sort(key=lambda x: x[3])

        # Greedy pack into micro-batches: keep adding rows while
        # (rows_in_batch + 1) * max_seq_len_in_batch <= budget.
        microbatches: list[list[tuple[int, list[int], list[int], int]]] = []
        current: list[tuple[int, list[int], list[int], int]] = []
        current_max = 0
        for item in keep:
            new_max = max(current_max, item[3])
            if current and (len(current) + 1) * new_max > budget:
                microbatches.append(current)
                current = [item]
                current_max = item[3]
            else:
                current.append(item)
                current_max = new_max
        if current:
            microbatches.append(current)

        for mb in microbatches:
            full_seqs = [p + a for _, p, a, _ in mb]
            max_len = max(len(s) for s in full_seqs)

            input_ids = torch.full((len(full_seqs), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(full_seqs), max_len), dtype=torch.long, device=device)
            for i, s in enumerate(full_seqs):
                input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
                attention_mask[i, : len(s)] = 1

            if use_ref:
                # If a SFT snapshot was registered as KL ref, swap its LoRA tensors
                # in for this no_grad forward, then restore. Otherwise fall back to
                # the C2 default of disabling LoRA entirely.
                if self._ref_lora_snapshot is not None:
                    with torch.no_grad():
                        saved: dict[str, dict[str, "torch.Tensor"]] = {}
                        for mod_name, snap in self._ref_lora_snapshot.items():
                            try:
                                mod = self.policy.model.get_submodule(mod_name)
                            except AttributeError:
                                continue
                            saved[mod_name] = {
                                "lora_A": mod.lora_A["default"].weight.data.clone(),
                                "lora_B": mod.lora_B["default"].weight.data.clone(),
                            }
                            mod.lora_A["default"].weight.data.copy_(snap["lora_A"])
                            mod.lora_B["default"].weight.data.copy_(snap["lora_B"])
                        try:
                            logits = self.policy.model(input_ids, attention_mask=attention_mask).logits
                        finally:
                            for mod_name, live in saved.items():
                                try:
                                    mod = self.policy.model.get_submodule(mod_name)
                                except AttributeError:
                                    continue
                                mod.lora_A["default"].weight.data.copy_(live["lora_A"])
                                mod.lora_B["default"].weight.data.copy_(live["lora_B"])
                else:
                    with torch.no_grad(), self.policy.model.disable_adapter():
                        logits = self.policy.model(input_ids, attention_mask=attention_mask).logits
            else:
                with torch.set_grad_enabled(True):
                    logits = self.policy.model(input_ids, attention_mask=attention_mask).logits

            for row, (orig_idx, prompt_ids, action_ids, _) in enumerate(mb):
                start = len(prompt_ids) - 1
                end = start + len(action_ids)
                slice_logits = logits[row, start:end, :].to(torch.float32)
                log_probs = F.log_softmax(slice_logits, dim=-1)
                target = torch.tensor(action_ids, dtype=torch.long, device=device)
                lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                out[orig_idx] = lp.detach() if use_ref else lp

            # Drop big tensors before the next micro-batch.
            del input_ids, attention_mask, logits

        return out

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
        # Collect all (prompt, action) pairs first; the heavy forward passes
        # (one new, one ref) happen ONCE per group via _batched_logprobs (M6).
        pa_pairs: list[tuple[list[int], list[int]]] = []
        per_turn_meta: list[tuple[int, int]] = []  # (i_traj, t_turn)
        for i, traj in enumerate(group.trajectories):
            for t, turn in enumerate(traj.turns):
                ids = list(turn.action_token_ids)
                prompt_ids = list(turn.prompt_token_ids)
                if not ids or not prompt_ids:
                    continue
                pa_pairs.append((prompt_ids, ids))
                per_turn_meta.append((i, t))

        if not pa_pairs:
            zero = torch.zeros((), device=device, requires_grad=True)
            return zero, TrainStepStats()

        new_lps = self._batched_logprobs(pa_pairs, use_ref=False)
        ref_lps = self._batched_logprobs(pa_pairs, use_ref=True)

        for k, ((prompt_ids, ids), (i, t)) in enumerate(zip(pa_pairs, per_turn_meta)):
            old_lp = torch.tensor(
                group.trajectories[i].turns[t].action_token_logprobs,
                dtype=torch.float32,
                device=device,
            )
            a_t = combined[i][t]
            adv_vec = torch.full((len(ids),), float(a_t), dtype=torch.float32, device=device)
            all_new_lp.append(new_lps[k])
            all_old_lp.append(old_lp)
            all_ref_lp.append(ref_lps[k])
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
        # KL warmup: zero out the term during the first N steps but still
        # observe + log it so the controller has data when warmup ends.
        if self._step < int(self.cfg.kl_warmup_episodes):
            kl_term = 0.0 * kl_per_tok.mean()
        else:
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
