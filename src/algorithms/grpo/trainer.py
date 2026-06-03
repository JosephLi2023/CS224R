"""HGPOTrainer: PPO-clipped policy update over K-trajectory groups, with the
H-GRPO advantage construction and an adaptive KL controller
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
    # H-GRPO advantage knobs
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
    # Zero out the KL penalty (still observed/logged) for the first N
    # train_step calls. Useful right after SFT warm-start when the k3
    # estimator can blow up. Default 0 = no warmup.
    kl_warmup_episodes: int = 0
    # When True, recompute new-policy logprobs in this trainer; when False the
    # caller passes them in (useful for unit tests with mocked policies).
    recompute_logprobs: bool = True
    # Cap on total tokens per padded forward in `_batched_logprobs`, to keep
    # activation memory bounded when KxT padded sequences would OOM the A100
    # shared with vLLM's KV cache. Default 2048.
    max_tokens_per_microbatch: int = 2048
    # Method B (TurnRD) integration knobs.
    # Re-load the TurnRD checkpoint every N rollout groups (0 disables).
    refresh_every_episodes: int = 0
    # Separate (higher) AdamW lr for the TurnRD params vs the LoRA policy.
    turnrd_lr: float = 1e-4
    # V-baseline annealing across rounds. V_theta is random-init noise in
    # Round 0, so ramp its weight in linearly:
    # beta = clamp(round_idx / warmup_rounds, 0, 1). Round 0 -> beta=0
    # (turn_adv only), warmup_rounds+ -> beta=1 (full V baseline).
    v_baseline_round_idx: int = 0
    v_baseline_warmup_rounds: int = 2
    # Proposal-A: drop alpha and project V_t onto the sum-to-R constraint
    # instead of the alpha*R formula:
    #   per_turn_t = (V_t_clamped - (sum_t V_t_clamped - R) / T_active) * mask
    # sum_t per_turn = R, but per_turn may be negative or exceed R (more
    # expressive than the [0,R] softmax-alpha form). Default False = legacy.
    use_v_projection_for_decomposition: bool = False
    v_projection_clamp: float = 2.0


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
    cls_query_norm: float = 0.0      # norm(cls_query); non-trivial change -> TurnRD is moving
    alpha_mean: float = 0.0          # mean of alpha over (k, t) entries
    alpha_var: float = 0.0           # variance of alpha; ~= uniform-flat when small
    alpha_max: float = 0.0           # max alpha weight in the group
    alpha_entropy: float = 0.0       # mean H(alpha); uniform on T turns -> log(T); peaked -> small
    alpha_progress_corr: float = 0.0 # mean Pearson corr(alpha, env raw_env_reward) over trajectories
    # Diagnostics for gradient-bearing signal magnitude.
    # The legacy `mean_traj_adv` / `mean_turn_adv` columns are mathematically
    # zero by construction (mean over centered values within a K-group).
    # The columns below carry the actual gradient-bearing signal magnitude.
    std_reward_group: float = 0.0    # std over the K final rewards; 0 -> degenerate K-group (all K rollouts gave same R)
    dead_K_group: int = 0            # 1 if std_reward_group < 1e-12 (no policy gradient on this group); else 0
    mean_abs_traj_adv: float = 0.0   # mean |A_traj| over K; non-zero whenever std_reward_group > 0
    std_traj_adv: float = 0.0        # std of A_traj over K; ~= 1 by construction whenever std_reward_group > 0
    mean_abs_adv_token: float = 0.0  # mean |adv_t| over the action tokens fed to PPO surrogate - the scalar gating policy_loss magnitude


PerTurnDecomposer = Callable[[TrajectoryGroup], list[list[float]]]
"""A turn decomposer: takes a TrajectoryGroup, returns list[K] of list[T_i]
of per-turn rewards r_hat_t^i (Methods A/B/C). For Method C (Progress) the
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

    One train_step minimizes:

        L = L_PPO(theta) + beta_t * KL_k3(pi_theta || pi_ref)
                         + lambda * L_consistency(theta, phi)

    - theta = LoRA-adapter weights (the only trainable LLM weights; base is
      frozen).
    - phi   = TurnRD decomposer params, trainable only for Method B
      (decomposer.has_learnable_params). Empty for Methods A/C.
    - pi_ref = LoRA-disabled base or an SFT-warmed LoRA snapshot; always
      constant w.r.t. theta and phi.

    L_PPO and KL flow into theta via new_logp (grad-on forward); L_PPO (via
    A_H) and L_consistency flow into phi via alpha_t. The two param groups
    use separate AdamW optimizers (different lrs) but share one backward.
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

        decomposer may be a plain callable (Methods A/C) or a TurnRDDecomposer
        object (Method B); the learnable surface is detected via
        getattr(decomposer, "has_learnable_params", False).

        refresh_decomposer_fn (optional) is invoked at
        cfg.refresh_every_episodes cadence inside train_step (skipping step 0)
        to reload the TurnRD checkpoint. It MUST mutate the decomposer's params
        in place (e.g. load_state_dict), not reassign decomposer.model: the
        second AdamW is built once over decomposer.parameters(), so a model
        swap would leave it updating stale tensors. With grad_accum_steps > 1,
        pick refresh_every_episodes as a multiple of grad_accum_steps so a
        refresh never lands mid-accumulation.
        """
        self.policy = policy
        self.decomposer = decomposer
        self.cfg = cfg or HGPOTrainerConfig()
        self.kl_controller = AdaptiveKLController(self.cfg.kl_cfg)
        self._optimizer: Any = None  # lazy-init in train_step
        self._step: int = 0
        # backward() calls since the last optimizer.step() flush. Keys the
        # grad-accum boundary off the real number of accumulated grads
        # (independent of self._step) so dead-K early returns (which skip
        # backward) can't misalign the boundary parity.
        self._pending_backwards: int = 0
        # Optional LoRA snapshot used as the KL reference: ref forwards swap
        # these in (no_grad) and restore live weights after. None -> the C2
        # frozen-base path (LoRA disabled). Set via snapshot_current_lora_as_ref().
        self._ref_lora_snapshot: dict[str, dict[str, Any]] | None = None
        # Method B: detect a learnable (TurnRD) decomposer. Methods A/C
        # inherit the False default via getattr.
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
        base Qwen - which would otherwise blow up the k3 estimator).
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

    # Pure-Python advantage stage (works without torch; used by tests)

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

    # Torch path (Modal A100 only; trainer.train_step)

    def _ensure_optimizer(self) -> None:
        if self._optimizer is not None:
            return
        import torch  # type: ignore[import-not-found]
        # Snapshot trainable params once so optimizer/grad-clip/AdamW state
        # can't drift if trainable_parameters() later returns a different
        # set/order (e.g. after merge_and_unload). Review M7.
        self._trainable_params: list = list(self.policy.trainable_parameters())
        self._optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=self.cfg.learning_rate,
        )
        # Separate AdamW for the TurnRD params when learnable, so each group
        # uses its own lr without sharing AdamW state.
        if self._decomposer_learnable and self._decomposer_optimizer is None:
            decomposer_params = list(self.decomposer.parameters())
            if decomposer_params:
                self._decomposer_optimizer = torch.optim.AdamW(
                    decomposer_params,
                    lr=self.cfg.turnrd_lr,
                )
                self._decomposer_params: list = decomposer_params
            else:
                # Learnable decomposer with zero params (shouldn't happen):
                # skip the optimizer so AdamW doesn't error on an empty group.
                self._decomposer_params = []

    def _new_logprobs_for_turn(
        self, prompt_token_ids: list[int], action_token_ids: list[int]
    ) -> "torch.Tensor":
        """Recompute per-token logprobs under the current policy for one
        (prompt, action) pair. Returns a 1-D tensor of len(action_token_ids).

        Runs grad-on with the LoRA active, so the result carries autograd
        history back to theta. fp32 before log_softmax keeps exp(new-old)
        stable in the importance ratio.
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
        # Cast to fp32 before log_softmax: bf16 logprob noise gets amplified
        # by exp() in the importance ratio (review M3).
        slice_logits = logits[0, start:end, :].to(torch.float32)
        log_probs = F.log_softmax(slice_logits, dim=-1)
        target = torch.tensor(action_token_ids, dtype=torch.long, device=device)
        return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [Tk]

    def _ref_logprobs_for_turn(
        self, prompt_token_ids: list[int], action_token_ids: list[int]
    ) -> "torch.Tensor":
        """Per-token logprobs under the frozen reference policy (LoRA disabled).

        Used for the KL-to-reference penalty (review C2). Runs under no_grad
        with disable_adapter() and returns detached tensors (constants in the
        loss graph).
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
        # fp32 cast (review M3); see _new_logprobs_for_turn for rationale.
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
        """Padded forward -> per-turn action logprobs (review M6).

        Splits the KxT (prompt, action) pairs into micro-batches under
        cfg.max_tokens_per_microbatch, releasing activations between them.

        use_ref=False: grad-on with LoRA active; returned tensors carry
        autograd into theta and drive L_PPO and the KL term.
        use_ref=True: no_grad with disable_adapter() (frozen base) or the SFT
        LoRA snapshot swapped in; returned tensors are detached. Live LoRA is
        restored in a finally block so an OOM mid-swap can't corrupt state.
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
                        # try opened before the swap so a partial swap (shape
                        # mismatch / OOM) still restores overwritten modules;
                        # saved[mod_name] is set before each copy_. (Review A2.)
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

        Reads prompt_token_ids / action_token_ids from each TurnRecord.

        Per action token:
            rho   = exp(new_logp - old_logp)
            L_PPO = -mean(min(rho * A_H, clip(rho, 1+-eps) * A_H))
            r     = exp(ref_logp - new_logp)
            KL_k3 = mean((r - 1) - log r)
            A_H   = alpha * A_traj + (1 - alpha) * A_turn

        theta (LoRA) gets gradient from L_PPO and KL via new_logp. phi (TurnRD,
        when learnable) gets it via alpha_t -> A_turn -> A_H -> L_PPO and via
        consistency_loss_tensor. pi_ref is never updated.

        KL warmup: during the first cfg.kl_warmup_episodes calls kl_term is a
        fresh zero tensor (not 0.0 * kl, which would propagate NaN/Inf);
        observed_kl is still logged.

        Returns (total_loss_tensor, TrainStepStats); stats are detached.
        """
        import torch  # type: ignore[import-not-found]

        adv = self.build_advantages(group)
        combined: list[list[float]] = adv["combined"]
        # Hoist traj_adv + flat_turn_adv so the V-head override block below
        # can reference traj_adv without UnboundLocalError.
        traj_adv: list[float] = adv["traj_adv"]
        flat_turn_adv: list[float] = [v for row in adv["turn_adv"] for v in row]

        # Signal diagnostics: mean_traj_adv/mean_turn_adv are zero by
        # construction, so these capture actual gradient signal instead:
        #   std_reward_group  = std of K final rewards (0 -> dead K-group)
        #   dead_K_group      = 1 if all K rollouts agree on R
        #   mean_abs_traj_adv = mean per-trajectory advantage magnitude
        #   std_traj_adv      = should be ~=1 (sanity check on normalisation)
        _final_R = group.final_rewards()
        if _final_R:
            _R_mean = sum(_final_R) / len(_final_R)
            _R_std = (sum((r - _R_mean) ** 2 for r in _final_R) / len(_final_R)) ** 0.5
        else:
            _R_std = 0.0
        _std_reward_group: float = float(_R_std)
        _dead_K_group: int = 1 if _std_reward_group < 1e-12 else 0
        if traj_adv:
            _mean_abs_traj_adv: float = sum(abs(a) for a in traj_adv) / len(traj_adv)
            _traj_mean = sum(traj_adv) / len(traj_adv)
            _std_traj_adv: float = (
                sum((a - _traj_mean) ** 2 for a in traj_adv) / len(traj_adv)
            ) ** 0.5
        else:
            _mean_abs_traj_adv = 0.0
            _std_traj_adv = 0.0

        device = next(self.policy.model.parameters()).device

        # Skip optimizer work for groups with no reward variation. The
        # decomposer has already run through `build_advantages`, so replay
        # emission and diagnostics still see the same inputs as live groups.
        if _dead_K_group:
            # Recover alpha statistics from the latest decomposer call when available.
            _alpha_mean = 0.0
            _alpha_var = 0.0
            _alpha_max = 0.0
            _alpha_entropy = 0.0
            _alpha_progress_corr = 0.0
            if (
                self._decomposer_learnable
                and getattr(self.decomposer, "_last_alpha", None) is not None
            ):
                try:
                    _last_alpha = self.decomposer._last_alpha
                    _last_mask = self.decomposer._last_alpha_mask
                    _last_idx = self.decomposer._last_alpha_traj_indices
                    if (
                        _last_alpha is not None
                        and _last_mask is not None
                        and _last_alpha.numel() > 0
                    ):
                        _denom2 = _last_mask.sum().clamp_min(1.0)
                        _alpha_mean = float(((_last_alpha * _last_mask).sum() / _denom2).item())
                        _alpha_var = float(
                            (((_last_alpha - _alpha_mean) ** 2 * _last_mask).sum() / _denom2).item()
                        )
                        # Guard masked_fill+max against the all-padded case
                        # (mask.sum()==0) where .max() would be -inf and leak
                        # into stats; keep the 0.0 sentinel instead.
                        if float(_last_mask.sum().item()) > 0.0:
                            _masked2 = _last_alpha.masked_fill(_last_mask == 0, float("-inf"))
                            _alpha_max = float(_masked2.max().item())
                        _log_alpha2 = torch.log(_last_alpha.clamp_min(1e-12))
                        _row_H2 = -(_last_alpha * _log_alpha2 * _last_mask).sum(dim=-1)
                        _alpha_entropy = float(_row_H2.mean().item())
                        _ac_sum = 0.0
                        _ac_n = 0
                        for _row, _orig_i in enumerate(_last_idx):
                            traj_obj = group.trajectories[_orig_i]
                            progress = [float(turn.raw_env_reward) for turn in traj_obj.turns]
                            if not progress or sum(abs(p) for p in progress) < 1e-12:
                                continue
                            a_row = _last_alpha[_row, :len(progress)].tolist()
                            if len(a_row) < 2:
                                continue
                            _n_t = len(progress)
                            _m_a = sum(a_row) / _n_t
                            _m_p = sum(progress) / _n_t
                            _cov = sum(
                                (a_row[k] - _m_a) * (progress[k] - _m_p) for k in range(_n_t)
                            )
                            _v_a = sum((a - _m_a) ** 2 for a in a_row)
                            _v_p = sum((p - _m_p) ** 2 for p in progress)
                            _den = (_v_a * _v_p) ** 0.5
                            if _den < 1e-12:
                                continue
                            _ac_sum += _cov / _den
                            _ac_n += 1
                        _alpha_progress_corr = (
                            _ac_sum / _ac_n if _ac_n > 0 else 0.0
                        )
                except Exception:
                    # Stats path must never break training.
                    pass
            # Normalize by hidden size for comparability across decomposer widths.
            _cls_query_norm = 0.0
            if self._decomposer_learnable:
                try:
                    _raw = float(
                        self.decomposer.model.cls_query.detach().norm().item()
                    )
                    _hidden = int(self.decomposer.model.cfg.hidden_size)
                    _cls_query_norm = _raw / max(1.0, _hidden ** 0.5)
                except AttributeError:  # pragma: no cover
                    _cls_query_norm = 0.0
            # Return a no-grad zero loss so the caller can skip backward().
            zero = torch.zeros((), device=device, dtype=torch.float32)
            stats = TrainStepStats(
                policy_loss=0.0,
                kl_term=0.0,
                consistency=float(adv["consistency"]),
                consistency_t=0.0,
                total_loss=0.0,
                observed_kl=0.0,
                kl_coef=self.kl_controller.coef,
                grad_norm=0.0,
                turnrd_grad_norm=0.0,
                n_action_tokens=0,
                mean_traj_adv=(sum(traj_adv) / max(1, len(traj_adv))),
                mean_turn_adv=(sum(flat_turn_adv) / max(1, len(flat_turn_adv))),
                cls_query_norm=_cls_query_norm,
                alpha_mean=_alpha_mean,
                alpha_var=_alpha_var,
                alpha_max=_alpha_max,
                alpha_entropy=_alpha_entropy,
                alpha_progress_corr=_alpha_progress_corr,
                std_reward_group=_std_reward_group,
                dead_K_group=1,
                mean_abs_traj_adv=_mean_abs_traj_adv,
                std_traj_adv=_std_traj_adv,
                mean_abs_adv_token=0.0,
            )
            return zero, stats

        all_new_lp: list[torch.Tensor] = []
        all_old_lp: list[torch.Tensor] = []
        all_ref_lp: list[torch.Tensor] = []
        # Assemble advantages after decomposer updates so any per-turn value
        # projection is reflected in the policy loss. The model forward
        # passes are batched once per group for efficiency.
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

        # Build only the logprob tensors here (they don't depend on
        # `combined`); all_adv is built after the grad block below.
        for k, ((prompt_ids, ids), (i, t)) in enumerate(zip(pa_pairs, per_turn_meta)):
            old_lp = torch.tensor(
                group.trajectories[i].turns[t].action_token_logprobs,
                dtype=torch.float32,
                device=device,
            )
            all_new_lp.append(new_lps[k])
            all_old_lp.append(old_lp)
            all_ref_lp.append(ref_lps[k])

        if not all_new_lp:
            zero = torch.zeros((), device=device, requires_grad=True)
            return zero, TrainStepStats()

        new_lp_t = torch.cat(all_new_lp).to(torch.float32)
        old_lp_t = torch.cat(all_old_lp).to(torch.float32)
        ref_lp_t = torch.cat(all_ref_lp).to(torch.float32)

        cons = float(adv["consistency"])
        # Consistency regularization has no gradient for pure-Python
        # decomposers. For learnable TurnRD, construct a tensor-valued loss
        # from grad-tracking attention weights so gradients can reach TurnRD.
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
                # Mask-weighted alpha stats for diagnostics: uniform ->
                # alpha_var ~= 0, alpha_max ~= 1/T; learned -> larger both.
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
                # Per-row entropy of alpha over unmasked positions, averaged
                # over K_real rows. Uniform on T_real -> log(T_real); smaller
                # = credit concentrated on fewer turns. Inline to avoid
                # importing from src.turnrd.
                _log_alpha = torch.log(alpha_t.clamp_min(1e-12))
                _row_H = -(alpha_t * _log_alpha * _mask_for_stats).sum(dim=-1)  # [K_real]
                _alpha_entropy = float(_row_H.mean().detach().item())
                # Per-turn rewards = alpha * R (legacy default); the
                # v-projection mode below can replace this. Then
                # group-normalize per turn position and per group, over the
                # K_real subset (empty trajectories are dropped, matching the
                # pure-Python path).
                R_for_decomp = final_R_t

                # v-projection mode (use_v_projection_for_decomposition):
                # replace alpha*R with a V_t projection onto sum-to-R:
                #   per_turn_t = (V_t_clamped - (sum V_t_clamped - R)/T_active) * mask
                # sum_t per_turn = R; per_turn may be negative or exceed R.
                # alpha is still computed (drives diagnostics + R-pred loss).
                if (
                    self.cfg.use_v_projection_for_decomposition
                    and grad_out.get("value_per_turn") is not None
                ):
                    v_t_raw = grad_out["value_per_turn"].detach()
                    if v_t_raw.shape == alpha_t.shape:
                        proj_mask = attn_t.to(dtype=v_t_raw.dtype)
                        v_t_clamped = v_t_raw.clamp(
                            -self.cfg.v_projection_clamp,
                            self.cfg.v_projection_clamp,
                        ) * proj_mask
                        # T_active = number of unmasked turns per row.
                        T_active = proj_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                        v_sum = v_t_clamped.sum(dim=-1, keepdim=True)
                        # Lagrangian projection: shift each turn by
                        # (sum V_t - R)/T_active so the new sum equals R.
                        adjustment = (v_sum - final_R_t.unsqueeze(-1)) / T_active
                        # Apply adjustment ONLY to active (unmasked) turns;
                        # padded turns stay at 0.
                        per_turn_rewards = (v_t_clamped - adjustment) * proj_mask
                    else:
                        per_turn_rewards = alpha_t * R_for_decomp.unsqueeze(-1)
                else:
                    per_turn_rewards = alpha_t * R_for_decomp.unsqueeze(-1)  # [K_real, T_max]
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
                    # K=1 -> group-normalised advantage is trivially 0.
                    traj_adv_t = torch.zeros_like(final_R_t)
                cons_loss_t = consistency_loss_tensor(
                    self.cfg.lambda_consistency, traj_adv_t, turn_adv_t, attn_t
                )
                # v6: use the V-head as a per-turn PPO baseline. When the
                # TurnRD model exposes value_per_turn, replace the normalized
                # turn_adv with an actor-critic baseline:
                #   A_t = alpha_t * R[i] - V_theta(h_t^i)
                # value is detached before going into `combined` (V is trained
                # by the standalone trainer, not policy backprop). Only
                # mask=1 entries are overridden.
                value_per_turn = grad_out.get("value_per_turn")
                if value_per_turn is not None and value_per_turn.shape == per_turn_rewards.shape:
                    # Per-position normalize the V-baseline before inserting
                    # into combined: raw (alpha*R - V) is ~0.03 while traj_adv
                    # is ~1.0, so unnormalized V would be effectively ignored.
                    v6_unnorm = (per_turn_rewards - value_per_turn) * mask_f
                    # Per-position group statistics, mask-aware (matches
                    # the existing turn_adv normalization path).
                    v6_col_mean = (v6_unnorm * mask_f).sum(dim=0) / col_count
                    v6_col_var = (
                        ((v6_unnorm - v6_col_mean.unsqueeze(0)) ** 2 * mask_f).sum(dim=0)
                        / col_count
                    )
                    v6_col_std = (v6_col_var + 1e-16).sqrt()
                    v6_turn_adv = (
                        (v6_unnorm - v6_col_mean.unsqueeze(0))
                        / v6_col_std.unsqueeze(0)
                    ) * mask_f  # [K_real, T_max], normalized
                    # Anneal the V-baseline by round: beta=0 in Round 0 (V is
                    # fresh-init noise -> old turn_adv only), beta=1 at
                    # warmup_rounds+, linear in between.
                    warmup = max(1, int(self.cfg.v_baseline_warmup_rounds))
                    beta_v = min(
                        1.0,
                        max(0.0, float(self.cfg.v_baseline_round_idx) / warmup),
                    )
                    # Only override when beta_v > 0: at beta=0 keep combined's
                    # original eval-mode values (falling through would swap in
                    # the dropout-perturbed train-mode advantages from
                    # decompose_with_grad). Review M1.
                    if beta_v > 0.0:
                        if beta_v >= 1.0:
                            v6_blended = v6_turn_adv  # full V baseline
                        else:
                            v6_blended = (
                                beta_v * v6_turn_adv
                                + (1.0 - beta_v) * turn_adv_t
                            )
                        v6_detached = v6_blended.detach().cpu().tolist()
                        nonempty_idx = grad_out["nonempty_indices"]
                        # Override combined[i][t] for unmasked entries, using
                        # combined[i][t] = alpha*traj_adv[i] + (1-alpha)*turn_adv.
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
                # Diagnostic: correlation between learned alpha and the env
                # progress signal. High -> TurnRD rediscovering Method C.
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

        # Populate alpha diagnostics from the eval-mode decomposition when
        # the gradient-tracking path is skipped by lambda_consistency == 0.
        # Only run when the gated block did not already populate them.
        if (
            self._decomposer_learnable
            and _alpha_mean == 0.0
            and _alpha_var == 0.0
            and getattr(self.decomposer, "_last_alpha", None) is not None
        ):
            try:
                _last_alpha = self.decomposer._last_alpha       # CPU [K_real, T_max]
                _last_mask = self.decomposer._last_alpha_mask    # CPU [K_real, T_max]
                _last_idx = self.decomposer._last_alpha_traj_indices  # list[int]
                if _last_alpha is not None and _last_mask is not None and _last_alpha.numel() > 0:
                    _denom2 = _last_mask.sum().clamp_min(1.0)
                    _alpha_mean = float(((_last_alpha * _last_mask).sum() / _denom2).item())
                    _alpha_var = float(
                        (((_last_alpha - _alpha_mean) ** 2 * _last_mask).sum() / _denom2).item()
                    )
                    _masked2 = _last_alpha.masked_fill(_last_mask == 0, float("-inf"))
                    _alpha_max = float(_masked2.max().item())
                    _log_alpha2 = torch.log(_last_alpha.clamp_min(1e-12))
                    _row_H2 = -(_last_alpha * _log_alpha2 * _last_mask).sum(dim=-1)
                    _alpha_entropy = float(_row_H2.mean().item())
                    # Pearson corr(alpha, raw_env_reward) per trajectory, then mean.
                    _ac_sum = 0.0
                    _ac_n = 0
                    for _row, _orig_i in enumerate(_last_idx):
                        traj_obj = group.trajectories[_orig_i]
                        progress = [float(turn.raw_env_reward) for turn in traj_obj.turns]
                        if not progress or sum(abs(p) for p in progress) < 1e-12:
                            continue
                        a_row = _last_alpha[_row, :len(progress)].tolist()
                        if len(a_row) < 2:
                            continue
                        _n_t = len(progress)
                        _m_a = sum(a_row) / _n_t
                        _m_p = sum(progress) / _n_t
                        _cov = sum((a_row[k] - _m_a) * (progress[k] - _m_p) for k in range(_n_t))
                        _v_a = sum((a - _m_a) ** 2 for a in a_row)
                        _v_p = sum((p - _m_p) ** 2 for p in progress)
                        _den = (_v_a * _v_p) ** 0.5
                        if _den < 1e-12:
                            continue
                        _ac_sum += _cov / _den
                        _ac_n += 1
                    _alpha_progress_corr = _ac_sum / _ac_n if _ac_n > 0 else 0.0
            except Exception:
                # Stats path must never break training. Leave alpha_* at 0
                # if anything went wrong recovering them.
                pass

        # Build the advantage tensor that feeds the PPO surrogate, reading
        # the (possibly V-head-overridden) `combined`.
        all_adv: list[torch.Tensor] = []
        for k, ((prompt_ids, ids), (i, t)) in enumerate(zip(pa_pairs, per_turn_meta)):
            a_t = combined[i][t]
            adv_vec = torch.full(
                (len(ids),), float(a_t), dtype=torch.float32, device=device
            )
            all_adv.append(adv_vec)
        adv_t = torch.cat(all_adv).to(torch.float32)

        # Scalar magnitude of the token-level advantage used in the policy loss.
        _mean_abs_adv_token: float = (
            float(adv_t.abs().mean().detach().item()) if adv_t.numel() > 0 else 0.0
        )

        # PPO importance ratio uses (new vs old=rollout) - that's correct.
        ratio = torch.exp(new_lp_t - old_lp_t)
        clip_eps = self.cfg.clip_eps
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        unclipped = ratio * adv_t
        policy_loss = -torch.minimum(unclipped, clipped).mean()

        # KL penalty: Schulman-k3 of KL(pi||pi_ref) needs r = pi_ref/pi
        # (log_ratio = ref_lp - new_lp), so E_pi[(r-1) - log r] = KL by Jensen
        # and each per-token term is non-negative. (Review A1.)
        ref_log_ratio = ref_lp_t - new_lp_t
        ref_ratio = torch.exp(ref_log_ratio)
        kl_per_tok = (ref_ratio - 1.0) - ref_log_ratio
        observed_kl = float(kl_per_tok.mean().detach().item())
        kl_coef = self.kl_controller.coef
        # KL warmup: zero the term for the first N steps with a FRESH tensor
        # (not 0.0 * kl.mean(), which would propagate NaN/Inf on overflow).
        # observed_kl is still logged. (Review A3 + A4.)
        if self._step < int(self.cfg.kl_warmup_episodes):
            kl_term = torch.zeros((), device=device, dtype=torch.float32)
        else:
            kl_term = kl_coef * kl_per_tok.mean()

        # Snapshot cons_loss_t (the gradient-bearing C3 loss) before the
        # optimizer step; distinct from `cons` (always 0 by construction).
        _cons_t_scalar: float = (
            float(cons_loss_t.detach().item()) if cons_loss_t is not None else 0.0
        )
        # cls_query norm, divided by sqrt(hidden_size) so it's comparable
        # across widths (init ~= 0.02 regardless of H; movement is the signal).
        _cls_query_norm: float = 0.0
        if self._decomposer_learnable:
            try:
                _raw = float(
                    self.decomposer.model.cls_query.detach().norm().item()
                )
                _hidden = int(self.decomposer.model.cfg.hidden_size)
                _cls_query_norm = _raw / max(1.0, _hidden ** 0.5)
            except AttributeError:  # pragma: no cover (decomposer without cls_query)
                _cls_query_norm = 0.0
        if cons_loss_t is not None:
            # .to(device) guards the test-only case where the decomposer is on
            # a different device than the policy; no-op when they match.
            total = policy_loss + kl_term + cons_loss_t.to(device=device)
        else:
            total = policy_loss + kl_term

        # `traj_adv` and `flat_turn_adv` were hoisted to the top of
        # compute_loss. They are still used here for stats only.
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
            std_reward_group=_std_reward_group,
            dead_K_group=_dead_K_group,
            mean_abs_traj_adv=_mean_abs_traj_adv,
            std_traj_adv=_std_traj_adv,
            mean_abs_adv_token=_mean_abs_adv_token,
        )
        return total, stats

    def train_step(self, group: TrajectoryGroup) -> TrainStepStats:
        """One AdamW step on a single TrajectoryGroup.

        Honors cfg.grad_accum_steps (loss divided by it; optimizer.step() only
        every Nth call). n_action_tokens == 0 short-circuits to a no-op.

        Order per call: optional in-place TurnRD checkpoint refresh ->
        _ensure_optimizer() -> compute_loss() builds
        L = L_PPO + beta*KL + lambda*L_cons -> (loss/accum).backward() ->
        at each grad-accum boundary, clip + step + zero_grad for the LoRA and
        (if learnable) the TurnRD optimizer -> KL controller update (frozen
        during warmup).

        self._step increments every call. Updated LoRA tensors are synced to
        vLLM by weight_sync.py and become pi_theta_old for the next rollout.
        """
        import torch  # type: ignore[import-not-found]

        # Refresh the TurnRD checkpoint at the configured cadence, before any
        # optimizer is built (it may swap the decomposer's model). Skipped at
        # step 0.
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
        if stats.dead_K_group:
            # Zero policy gradient by construction; compute_loss already
            # short-circuited, so skip backward + optimizer step. Don't feed
            # the controller observed_kl=0 either (a run of dead groups would
            # collapse kl_coef to min_coef); leave it untouched. _step still
            # advances so refresh + warmup counts stay aligned.
            stats.kl_coef = self.kl_controller.coef
            self._step += 1
            return stats
        if stats.n_action_tokens == 0:
            # No live action tokens this group; nothing to learn from.
            new_coef = self.kl_controller.update(stats.observed_kl)
            stats.kl_coef = new_coef
            self._step += 1
            return stats

        accum = max(1, int(self.cfg.grad_accum_steps))
        scaled_loss = loss / accum
        scaled_loss.backward()
        self._pending_backwards += 1

        # Step once we've accumulated `accum` backward contributions. Keying
        # off _pending_backwards (not _step) keeps dead-K early returns from
        # misaligning the boundary parity.
        is_step_boundary = self._pending_backwards >= accum
        if is_step_boundary:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_params, self.cfg.max_grad_norm
            )
            stats.grad_norm = float(grad_norm)
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            # Also step the TurnRD optimizer when present.
            if self._decomposer_optimizer is not None and self._decomposer_params:
                turnrd_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self._decomposer_params, self.cfg.max_grad_norm
                )
                stats.turnrd_grad_norm = float(turnrd_grad_norm)
                self._decomposer_optimizer.step()
                self._decomposer_optimizer.zero_grad(set_to_none=True)
            self._pending_backwards = 0
        else:
            stats.grad_norm = 0.0  # not yet a step boundary

        # Freeze the KL controller during warmup (review A3): a post-SFT KL
        # spike would otherwise saturate kl_coef before warmup ends.
        if self._step >= int(self.cfg.kl_warmup_episodes):
            new_coef = self.kl_controller.update(stats.observed_kl)
            stats.kl_coef = new_coef
        # else: kl_coef stays at controller.coef (init value), already in stats
        self._step += 1
        return stats
