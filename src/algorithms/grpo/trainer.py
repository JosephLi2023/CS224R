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
    consistency_loss_tensor,
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
    # H-GRPO Method B (TurnRD) integration knobs (Day 13).
    # Re-load the TurnRD checkpoint every N rollout groups (0 disables —
    # preserves prior behavior for Methods A/C and existing tests).
    refresh_every_episodes: int = 0
    # Separate AdamW learning rate for the TurnRD parameters. Kept distinct
    # from `learning_rate` (1e-6 for the LoRA policy) since TurnRD is a
    # small standalone Transformer that benefits from a higher rate (~1e-4
    # is the standalone trainer default).
    turnrd_lr: float = 1e-4


@dataclass
class TrainStepStats:
    policy_loss: float = 0.0
    kl_term: float = 0.0
    consistency: float = 0.0  # python-side advantage residual (always 0 by construction; kept for back-compat)
    consistency_t: float = 0.0  # tensor-side gradient-bearing C3 loss (the one that actually drives the trainer)
    total_loss: float = 0.0
    observed_kl: float = 0.0
    kl_coef: float = 0.0
    grad_norm: float = 0.0           # policy LoRA grad norm (pre-clip)
    turnrd_grad_norm: float = 0.0    # TurnRD parameter grad norm (pre-clip; 0 when not learnable)
    n_action_tokens: int = 0
    mean_traj_adv: float = 0.0
    mean_turn_adv: float = 0.0
    # TurnRD diagnostics (populated only when decomposer is learnable):
    cls_query_norm: float = 0.0      # ‖cls_query‖₂ — non-trivial change ⇒ TurnRD is moving
    alpha_mean: float = 0.0          # mean of α over (k, t) entries
    alpha_var: float = 0.0           # variance of α; ≈ uniform-flat when small
    alpha_max: float = 0.0           # max α weight in the group
    alpha_entropy: float = 0.0       # mean H(α); uniform on T turns ⇒ log(T); peaked ⇒ small
    alpha_progress_corr: float = 0.0 # mean Pearson corr(α, env raw_env_reward) over trajectories


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

    Gradient-flow overview
    ----------------------
    For one `train_step(group)` call the trainer minimizes::

        L = L_PPO(theta)  +  beta_t * KL_k3(pi_theta || pi_ref)
                          +  lambda * L_consistency(theta, phi)

    where:
      * ``theta``  = LoRA-adapter weights on the trainable Qwen body
        (the ONLY LLM weights that receive gradients; the base ``W_0`` is
        frozen).
      * ``phi``    = TurnRD decomposer parameters — present and trainable
        only when ``decomposer.has_learnable_params is True``
        (Method B). For Methods A/C this set is empty.
      * ``pi_ref`` = either the LoRA-disabled base (``disable_adapter()``)
        or a snapshot of an SFT-warmed LoRA captured by
        ``snapshot_current_lora_as_ref()`` — always a *constant* w.r.t.
        ``theta`` and ``phi``.

    The two distinct gradient paths in a single backward pass are:

    1. ``L_PPO`` and the KL term flow gradients into ``theta`` only,
       through the grad-on forward in ``_batched_logprobs(use_ref=False)``
       which produces ``new_logp(pi_theta)``. ``old_logp`` (rollout cache)
       and ``ref_logp`` (no_grad forward) are constants.
    2. ``L_PPO`` (via the broadcast advantage ``A_H``) AND ``L_consistency``
       (tensor form) flow gradients into ``phi`` through the grad-tracking
       ``alpha_t`` produced by ``decompose_with_grad`` →
       ``r_hat_t = alpha_t * R`` → ``A_turn`` → ``A_H``.

    The two parameter groups are stepped by **two separate AdamW
    optimizers** (different learning rates: ``cfg.learning_rate`` for the
    LoRA, ``cfg.turnrd_lr`` for TurnRD) so AdamW state is not shared.
    Both share a single ``loss.backward()`` call.
    """

    def __init__(
        self,
        policy: "LoRAPolicy",
        decomposer: PerTurnDecomposer,
        cfg: HGPOTrainerConfig | None = None,
        *,
        refresh_decomposer_fn: Callable[[], None] | None = None,
    ) -> None:
        """Construct an H-GRPO trainer.

        Args:
            policy: the trainable LoRA-wrapped policy.
            decomposer: a per-turn reward decomposer. May be a plain
                callable (Methods A/C) or a `TurnRDDecomposer` *object*
                (Method B); the trainer detects the learnable surface via
                `getattr(decomposer, "has_learnable_params", False)`.
            cfg: optional `HGPOTrainerConfig`; defaults are used if omitted.
            refresh_decomposer_fn: optional callable invoked at
                `cfg.refresh_every_episodes` cadence inside `train_step`
                (skipping `_step=0`). Intended for periodic re-loading of
                the TurnRD checkpoint from disk.

                Contract: the refresh fn MUST mutate the decomposer's
                parameters IN PLACE (e.g. via
                `decomposer.load_state_dict(...)`), not by reassigning
                `decomposer.model = new_TurnRD(...)`. The trainer's
                second AdamW is built once over the parameter tensors
                returned by `decomposer.parameters()`; an in-place load
                preserves those tensor identities, while a model swap
                would leave the optimizer holding stale references that
                update params nobody reads.

                Safety with `grad_accum_steps > 1`: pick
                `refresh_every_episodes` as a multiple of
                `grad_accum_steps` so the refresh never lands inside an
                in-progress gradient accumulation window (otherwise
                accumulated TurnRD grads from before the refresh would
                be applied to post-refresh parameters). With the default
                `grad_accum_steps=1` this is automatic.
        """
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
        # H-GRPO Method B integration (Day 13).
        # Detect whether the decomposer is a learnable Method B (TurnRD)
        # decomposer. Other decomposers (Methods A/C) inherit the False
        # default via `getattr` so they don't need to declare anything.
        self._decomposer_learnable: bool = bool(
            getattr(self.decomposer, "has_learnable_params", False)
        )
        self._refresh_fn: Callable[[], None] | None = refresh_decomposer_fn
        # Built lazily inside _ensure_optimizer (matches the policy
        # optimizer's lazy init); kept None for non-learnable decomposers.
        self._decomposer_optimizer: Any = None

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
        # Day 13: also build a separate AdamW for the TurnRD parameters
        # when the decomposer is learnable. Kept separate from the policy
        # optimizer so each param group can use its own lr (`turnrd_lr`
        # vs `learning_rate`) without sharing AdamW state.
        if self._decomposer_learnable and self._decomposer_optimizer is None:
            decomposer_params = list(self.decomposer.parameters())
            if decomposer_params:
                self._decomposer_optimizer = torch.optim.AdamW(
                    decomposer_params,
                    lr=self.cfg.turnrd_lr,
                )
                self._decomposer_params: list = decomposer_params
            else:
                # Edge case: a learnable decomposer with zero parameters
                # (shouldn't happen in production). Skip the optimizer so
                # AdamW doesn't error on an empty param group.
                self._decomposer_params = []

    def _new_logprobs_for_turn(
        self, prompt_token_ids: list[int], action_token_ids: list[int]
    ) -> "torch.Tensor":
        """Recompute per-token logprobs under the *current* policy for a single
        (prompt, action) pair. Returns a 1-D tensor of length len(action_token_ids).

        Gradient flow
        -------------
        Runs under ``torch.set_grad_enabled(True)`` with the LoRA adapter
        active, so the returned tensor carries autograd history back to
        ``theta`` (the LoRA params). This is the path through which
        ``L_PPO`` and ``KL_k3`` ultimately reach ``optimizer.step()``
        for the LLM weights. The fp32 cast before ``log_softmax`` keeps
        ``exp(new - old)`` numerically stable in the importance ratio.
        """
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

        Gradient flow
        -------------
        Runs under ``torch.no_grad()`` and returns ``.detach()``-ed
        tensors. The reference logprobs are *constants* in the loss
        graph — they only set the target distribution that ``KL_k3``
        penalizes drift from. No gradient ever flows into ``pi_ref``
        (it is, by construction, frozen).
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

        Gradient flow
        -------------
        ``use_ref`` selects between the two distinct forward modes used
        every train step:

        * ``use_ref=False`` — runs under ``torch.set_grad_enabled(True)``
          with the LoRA active. Returned tensors carry autograd history
          back to ``theta`` (the LoRA params). These ``new_logp`` values
          drive ``L_PPO`` (via the importance ratio ``rho = exp(new-old)``)
          AND the KL term (via the k3 estimator, where ``new_logp`` is the
          denominator inside ``r = pi_ref / pi_theta``).

        * ``use_ref=True`` — runs under ``torch.no_grad()`` either with
          ``disable_adapter()`` (default reference = frozen base) or with
          the SFT LoRA snapshot temporarily swapped in (when
          ``self._ref_lora_snapshot is not None``). Returned tensors are
          ``.detach()``-ed and act as constants in the loss graph. Live
          LoRA tensors are restored in a ``finally`` block so a partial
          swap on OOM cannot leave the policy in a corrupted state.

        Both modes return per-row 1-D tensors aligned with each
        (prompt, action) pair's action-token positions; padding rows have
        zero gradient contribution since they were filtered out before
        the micro-batch was assembled.
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
                        # Open try BEFORE the swap-in so a partial swap (e.g. shape
                        # mismatch / OOM mid-loop) still triggers restoration of
                        # the modules already overwritten. saved[mod_name] is
                        # populated *before* each copy_, so restoration is well-
                        # defined at any point. (Review item A2.)
                        try:
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

        Gradient flow (``L = L_PPO + beta_t * KL_k3 + lambda * L_cons``)
        ----------------------------------------------------------------
        Per action token ``u`` of turn ``t`` in trajectory ``i``::

            rho      = exp(new_logp - old_logp)         # autograd → theta
            L_PPO    = -mean( min(rho * A_H, clip(rho, 1±eps) * A_H) )
            r        = pi_ref / pi_theta = exp(ref_logp - new_logp)
            KL_k3    = mean( (r - 1) - log r )           # autograd → theta
            A_H      = alpha * A_traj + (1 - alpha) * A_turn

        Gradient destinations:

        * **theta (LoRA)** receives gradient from
          ``L_PPO`` (via ``new_logp``) and from ``KL_k3`` (via
          ``new_logp`` — note ``ref_logp`` is detached). ``old_logp`` is a
          constant from the rollout cache.

        * **phi (TurnRD)** — present only when
          ``self._decomposer_learnable is True``. Gradient flows in via
          ``decompose_with_grad`` → ``alpha_t`` (grad-tracking) →
          ``per_turn_rewards = alpha * R`` → group-normalized ``A_turn``
          → ``A_H`` → ``L_PPO``, AND additionally via the tensor-form
          ``consistency_loss_tensor`` (``L_cons``). ``A_traj`` is a
          constant w.r.t. ``phi`` because it depends only on the env
          rewards ``R_i``.

        * **pi_ref** is *never* updated — both reference paths
          (``disable_adapter()`` and the SFT snapshot swap) run under
          ``torch.no_grad()`` and the returned tensors are detached.

        KL warmup: during the first ``cfg.kl_warmup_episodes`` calls,
        ``kl_term`` is replaced by a fresh zero tensor (NOT ``0.0 *
        kl_per_tok.mean()`` — that would propagate NaN/Inf from a blown-
        up k3 estimator right after SFT). ``observed_kl`` is still
        recorded for logging.

        Returns
        -------
        total : ``torch.Tensor``
            Scalar loss with autograd history into both ``theta`` and
            ``phi`` (when learnable). Caller handles ``backward()``.
        stats : ``TrainStepStats``
            Detached scalars for logging — never participates in the
            gradient computation.
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

        # KL penalty uses (ref vs new) with samples from the current policy π.
        # For unbiased Schulman-k3 of KL(π||π_ref), the ratio must be r = π_ref/π
        # (i.e. log_ratio = ref_lp − new_lp), NOT r = π/π_ref. Then
        # E_π[(r − 1) − log r] = KL(π||π_ref) by Jensen, and the per-token term
        # is non-negative. (Review item A1: prior version used new − ref which
        # was a non-negative biased surrogate of neither KL direction.)
        ref_log_ratio = ref_lp_t - new_lp_t
        ref_ratio = torch.exp(ref_log_ratio)
        kl_per_tok = (ref_ratio - 1.0) - ref_log_ratio
        observed_kl = float(kl_per_tok.mean().detach().item())
        kl_coef = self.kl_controller.coef
        # KL warmup: zero out the term during the first N steps with a FRESH
        # tensor (not 0.0 * kl_per_tok.mean(), which would propagate NaN/inf
        # if ref_ratio overflowed — exactly the scenario warmup exists to
        # handle). observed_kl is still computed + logged so we can see what
        # the un-penalised KL is doing during warmup. (Review items A3 + A4.)
        if self._step < int(self.cfg.kl_warmup_episodes):
            kl_term = torch.zeros((), device=device, dtype=torch.float32)
        else:
            kl_term = kl_coef * kl_per_tok.mean()

        cons = float(adv["consistency"])
        # Consistency reg has zero gradient for Methods A/C (the pure-Python
        # `consistency_loss` returns a Python float, not a torch leaf, so the
        # original C3 fix removed it from `total`). For Method B (TurnRD),
        # which IS learnable, we re-add the regulariser as a torch tensor
        # built from the model's grad-tracking α weights — this gives it a
        # real gradient path back to TurnRD parameters per the C3 follow-up
        # note in `MEDIUM_FIXES.md::C3`.
        cons_loss_t: "torch.Tensor | None" = None
        # TurnRD diagnostics (populated alongside cons_loss_t when learnable).
        _alpha_mean: float = 0.0
        _alpha_var: float = 0.0
        _alpha_max: float = 0.0
        _alpha_entropy: float = 0.0
        _alpha_progress_corr: float = 0.0
        if self._decomposer_learnable and self.cfg.lambda_consistency != 0.0:
            grad_out = self.decomposer.decompose_with_grad(group)
            alpha_t = grad_out["alpha"]  # [K_real, T_max], grad-tracking
            attn_t = grad_out["attention_mask"]  # [K_real, T_max] long
            final_R_t = grad_out["final_R"]  # [K_real]
            if alpha_t.numel() > 0:
                # Snapshot α distribution stats (mask-weighted) for
                # post-hoc diagnostics: a uniform decomposition has
                # alpha_var ≈ 0 and alpha_max ≈ 1/T; learned decompositions
                # concentrate on a few turns and grow alpha_max + alpha_var.
                _mask_for_stats = attn_t.to(dtype=alpha_t.dtype)
                _denom = _mask_for_stats.sum().clamp_min(1.0)
                _alpha_mean = float(((alpha_t * _mask_for_stats).sum() / _denom).detach().item())
                _alpha_var = float(
                    (((alpha_t - _alpha_mean) ** 2 * _mask_for_stats).sum() / _denom).detach().item()
                )
                # Mask out padding positions before max so an all-zero
                # padded position doesn't artificially shrink the max.
                _masked = alpha_t.masked_fill(attn_t == 0, float("-inf"))
                _alpha_max = float(_masked.max().detach().item()) if alpha_t.numel() > 0 else 0.0
                # Per-row entropy of α over the unmasked positions, then
                # averaged across K_real rows. Uniform on T_real ⇒ log(T_real).
                # Smaller = α concentrating credit on fewer turns. We compute
                # it inline (rather than calling model.alpha_entropy) so the
                # trainer doesn't need to import from src.turnrd.
                _log_alpha = torch.log(alpha_t.clamp_min(1e-12))
                _row_H = -(alpha_t * _log_alpha * _mask_for_stats).sum(dim=-1)  # [K_real]
                _alpha_entropy = float(_row_H.mean().detach().item())
                # Per-turn rewards = α * R. Then group-normalize per
                # turn position (matches `compute_turn_advantages`) AND
                # group-normalize final_R per group (matches
                # `compute_traj_advantages`). Both norms operate on the
                # K_real subset (empty trajectories are dropped from the
                # tensor stack), which mirrors the pure-Python path's
                # behavior — empty trajectories contribute nothing to
                # either advantage's mean/std.
                per_turn_rewards = alpha_t * final_R_t.unsqueeze(-1)  # [K_real, T_max]
                # Per-position turn advantage normalization, mask-aware.
                # mask_f: [K_real, T_max] of {0.0, 1.0}.
                mask_f = attn_t.to(dtype=per_turn_rewards.dtype)
                # Per-position counts: [T_max].
                col_count = mask_f.sum(dim=0).clamp_min(1.0)
                col_mean = (per_turn_rewards * mask_f).sum(dim=0) / col_count
                col_var = ((per_turn_rewards - col_mean.unsqueeze(0)) ** 2 * mask_f).sum(dim=0) / col_count
                col_std = (col_var + 1e-16).sqrt()
                turn_adv_t = ((per_turn_rewards - col_mean.unsqueeze(0)) / col_std.unsqueeze(0)) * mask_f
                # Per-group trajectory advantage normalization.
                K_real = final_R_t.shape[0]
                if K_real > 1:
                    R_mean = final_R_t.mean()
                    R_var = ((final_R_t - R_mean) ** 2).mean()
                    R_std = (R_var + 1e-16).sqrt()
                    traj_adv_t = (final_R_t - R_mean) / R_std
                else:
                    # K=1 → group-normalised advantage is trivially 0.
                    traj_adv_t = torch.zeros_like(final_R_t)
                cons_loss_t = consistency_loss_tensor(
                    self.cfg.lambda_consistency, traj_adv_t, turn_adv_t, attn_t
                )
                # v6: V-head as per-turn PPO baseline. When the
                # TurnRD model has a learnable value head AND it
                # exposed value_per_turn here, REPLACE the
                # per-position-normalized turn_adv used in the PPO
                # surrogate with a true actor-critic baseline:
                #   A_t = (α_t · R[i]) − V_θ(h_t^i)
                # The value tensor is grad-tracking, but we DETACH
                # it before insertion into `combined` so the policy
                # gradient doesn't push V (V is trained via
                # standalone trainer's loss_value_head, not via
                # policy backprop). Only the (i, t) entries with
                # mask=1 are overridden; padded turns stay at 0.
                value_per_turn = grad_out.get("value_per_turn")
                if value_per_turn is not None and value_per_turn.shape == per_turn_rewards.shape:
                    v6_turn_adv = (
                        per_turn_rewards - value_per_turn
                    ) * mask_f  # [K_real, T_max], grad-tracking
                    v6_detached = v6_turn_adv.detach().cpu().tolist()
                    nonempty_idx = grad_out["nonempty_indices"]
                    # Override combined[i][t] for the unmasked entries.
                    # combined came from build_advantages (Python list); we
                    # rebuild only the entries we have V predictions for.
                    # combined[i][t] = α·traj_adv[i] + (1-α)·turn_adv[i][t]
                    alpha_w = float(self.cfg.alpha)
                    for row, orig_i in enumerate(nonempty_idx):
                        traj_obj = group.trajectories[orig_i]
                        T_i = len(traj_obj.turns)
                        for t in range(min(T_i, len(v6_detached[row]))):
                            if attn_t[row, t].item() == 0:
                                continue
                            new_turn_adv = v6_detached[row][t]
                            combined[orig_i][t] = (
                                alpha_w * traj_adv[orig_i] +
                                (1.0 - alpha_w) * new_turn_adv
                            )
                # v6 diagnostic: correlation between learned α and the
                # env's normalized progress signal (Method C's signal).
                # High correlation ⇒ TurnRD is rediscovering Method C.
                # Low ⇒ TurnRD has found a different signal.
                _alpha_corr_sum = 0.0
                _alpha_corr_n = 0
                for row, orig_i in enumerate(grad_out["nonempty_indices"]):
                    traj_obj = group.trajectories[orig_i]
                    progress = [float(turn.raw_env_reward) for turn in traj_obj.turns]
                    if not progress or sum(abs(p) for p in progress) < 1e-12:
                        continue
                    a_row = alpha_t[row, :len(progress)].detach().cpu().tolist()
                    if len(a_row) < 2:
                        continue
                    # Pearson correlation across turn positions.
                    n_t = len(progress)
                    mean_a = sum(a_row) / n_t
                    mean_p = sum(progress) / n_t
                    cov = sum((a_row[k] - mean_a) * (progress[k] - mean_p) for k in range(n_t))
                    var_a = sum((a - mean_a) ** 2 for a in a_row)
                    var_p = sum((p - mean_p) ** 2 for p in progress)
                    denom = (var_a * var_p) ** 0.5
                    if denom < 1e-12:
                        continue
                    _alpha_corr_sum += cov / denom
                    _alpha_corr_n += 1
                _alpha_progress_corr = (
                    _alpha_corr_sum / _alpha_corr_n if _alpha_corr_n > 0 else 0.0
                )
        # Snapshot the cons_loss_t scalar BEFORE the optimizer step. This
        # is the gradient-bearing C3 loss — distinct from `cons` (the
        # pure-Python residual which is always 0 by construction).
        _cons_t_scalar: float = (
            float(cons_loss_t.detach().item()) if cons_loss_t is not None else 0.0
        )
        # Snapshot cls_query norm — proves whether TurnRD is moving.
        _cls_query_norm: float = 0.0
        if self._decomposer_learnable:
            try:
                _cls_query_norm = float(
                    self.decomposer.model.cls_query.detach().norm().item()
                )
            except AttributeError:  # pragma: no cover (decomposer without cls_query)
                _cls_query_norm = 0.0
        if cons_loss_t is not None:
            # Guard against the (production-impossible but test-possible) case
            # where the decomposer lives on a different device than the policy.
            # On A100 production both end up on cuda:0; on CPU tests everything
            # is already on CPU. The .to() is a no-op when devices match.
            total = policy_loss + kl_term + cons_loss_t.to(device=device)
        else:
            total = policy_loss + kl_term

        traj_adv = adv["traj_adv"]
        flat_turn_adv: list[float] = [v for row in adv["turn_adv"] for v in row]
        stats = TrainStepStats(
            policy_loss=float(policy_loss.detach().item()),
            kl_term=float(kl_term.detach().item()),
            consistency=cons,
            consistency_t=_cons_t_scalar,
            total_loss=float(total.detach().item()),
            observed_kl=observed_kl,
            kl_coef=kl_coef,
            grad_norm=0.0,
            n_action_tokens=int(new_lp_t.numel()),
            mean_traj_adv=(sum(traj_adv) / max(1, len(traj_adv))),
            mean_turn_adv=(sum(flat_turn_adv) / max(1, len(flat_turn_adv))),
            cls_query_norm=_cls_query_norm,
            alpha_mean=_alpha_mean,
            alpha_var=_alpha_var,
            alpha_max=_alpha_max,
            alpha_entropy=_alpha_entropy,
            alpha_progress_corr=_alpha_progress_corr,
        )
        return total, stats

    def train_step(self, group: TrajectoryGroup) -> TrainStepStats:
        """One AdamW step on a single TrajectoryGroup.

        Honors `cfg.grad_accum_steps` (review M4): the loss is divided by
        `grad_accum_steps` and `optimizer.step()` is only called every Nth
        invocation. `n_action_tokens == 0` short-circuits to a no-op
        (review nit) so we don't crash on a leaf-tensor backward.

        Gradient-flow ordering (one invocation)
        ---------------------------------------
        1. Optional TurnRD checkpoint refresh (in-place; preserves
           tensor identities held by ``self._decomposer_optimizer``).
        2. ``_ensure_optimizer()`` lazily builds AdamW(theta_LoRA) and,
           when the decomposer is learnable, AdamW(phi_TurnRD).
        3. ``compute_loss(group)`` builds the autograd graph:

               L = L_PPO + beta * KL + lambda * L_cons

           * grad path 1: ``new_logp(pi_theta)`` → L_PPO + KL → theta
           * grad path 2: ``alpha_t(phi)`` → A_turn → A_H → L_PPO,
             plus ``alpha_t(phi)`` → L_cons → phi

        4. ``(loss / grad_accum_steps).backward()`` accumulates grads
           into ``.grad`` on both parameter groups. Scaling is applied
           BEFORE backward so AdamW state stays unscaled.
        5. **Step boundary** (every ``grad_accum_steps`` calls):

              - ``clip_grad_norm_(theta_LoRA, max_grad_norm)``
                then ``AdamW(theta_LoRA).step()`` + ``zero_grad()``
              - if learnable decomposer:
                ``clip_grad_norm_(phi_TurnRD, max_grad_norm)``
                then ``AdamW(phi_TurnRD).step()`` + ``zero_grad()``

           Otherwise grads keep accumulating silently and
           ``stats.grad_norm`` is set to 0.0.
        6. ``AdaptiveKLController.update(observed_kl)`` adjusts
           ``beta_{t+1}`` for the next group — frozen during KL warmup
           so a post-SFT KL spike doesn't saturate the controller.

        Side effects across calls
        -------------------------
        * ``self._step`` is incremented every call (used to gate KL
          warmup and decomposer-refresh cadence).
        * Updated LoRA tensors are picked up by
          ``src/policy/weight_sync.py`` on the next sync pass and
          pushed to vLLM, becoming the new ``pi_theta_old`` for the
          next rollout batch — closing the on-policy-ish PPO loop.
        """
        import torch  # type: ignore[import-not-found]

        # Day 13: refresh the TurnRD checkpoint at the configured cadence
        # BEFORE building any optimizer (the refresh may swap the
        # decomposer's underlying model; the optimizer must see the new
        # parameters). Skipped at step 0 so we don't refresh the brand-new
        # decomposer that was just constructed.
        if (
            self._refresh_fn is not None
            and self.cfg.refresh_every_episodes > 0
            and self._step > 0
            and (self._step % int(self.cfg.refresh_every_episodes) == 0)
        ):
            self._refresh_fn()

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
            # Day 13: also step the TurnRD optimizer when present.
            if self._decomposer_optimizer is not None and self._decomposer_params:
                turnrd_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._decomposer_params, self.cfg.max_grad_norm
                )
                stats.turnrd_grad_norm = float(turnrd_grad_norm)
                self._decomposer_optimizer.step()
                self._decomposer_optimizer.zero_grad(set_to_none=True)
        else:
            stats.grad_norm = 0.0  # not yet a step boundary

        # KL controller is also frozen during warmup (review item A3): if
        # observed_kl spikes to ~1e6 right after SFT, feeding the controller
        # would saturate kl_coef to floor/ceiling before warmup ends, snapping
        # the penalty on at an absurd coefficient. Resume updates AFTER warmup.
        if self._step >= int(self.cfg.kl_warmup_episodes):
            new_coef = self.kl_controller.update(stats.observed_kl)
            stats.kl_coef = new_coef
        # else: kl_coef stays at controller.coef (init value), already in stats
        self._step += 1
        return stats
