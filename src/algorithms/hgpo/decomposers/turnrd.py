"""TurnRD per-turn reward decomposer adapter (Method B; proposal §3.2).

Inference-only adapter that the trainer plugs in. The training loop
(replay-buffer reader + standalone train script + HGPOTrainer refresh hook)
ships Day 13–14 — see `~/.llms/plans/cs224r_hgpo_method_b_turnrd_m1.plan.md`
"What's deliberately NOT included" section.

torch is imported at the top of this file; the embedder callable is supplied
by the caller, so unit tests can drive the adapter with a deterministic
stub embedder that returns plain tensors — no LoRAPolicy or HF model needed.

The §3.2 invariant `Σ_t r̂_t = R` per trajectory holds by construction
(`TurnRDOutput.decompose(R) = α · R` where `Σ α = 1` after masking).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

import torch

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup
from src.turnrd.model import TurnRD


# Embedder contract: per-trajectory callable returning a per-turn embedding
# tensor of shape `[T_i, D]` (D == TurnRD.input_dim).
#
# Production wires this from `LoRAPolicy.model.eval()` mean-pooled hidden
# states (Day 14); tests pass a deterministic stub.
#
# Contract requirements (the adapter enforces (a)+(b) defensively, (c) is
# the embedder's responsibility but the adapter wraps the call in
# `torch.no_grad()` belt-and-suspenders so a forgetful caller doesn't leak
# memory):
#   (a) shape: `[T_i, D]` (asserted in `decompose`).
#   (b) device + dtype: free; the adapter casts to the model's parameter
#       device + dtype before forward (so the embedder MAY return e.g. CPU
#       fp32 tensors even when the model lives on a CUDA bf16 device).
#   (c) gradient: ideally `.detach()` or computed under `torch.no_grad()`,
#       since the adapter only needs the values. The adapter's `no_grad`
#       wrapper neutralises a careless implementation but does NOT free a
#       graph that already exists on tensors returned BEFORE the wrapper
#       takes effect — so production embedders SHOULD still detach.
TurnEmbedder = Callable[[Trajectory], torch.Tensor]


class TurnRDDecomposer:
    """Per-turn reward decomposer that delegates to a learned `TurnRD` model.

    Plugs into `HGPOTrainer(decomposer=TurnRDDecomposer(...).decompose)`.
    """

    def __init__(
        self,
        model: TurnRD,
        embedder: TurnEmbedder,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.embedder = embedder
        # Resolve the target device once. Default to the model's parameter
        # device so a caller doing `TurnRDDecomposer(cuda_model, cpu_embedder)`
        # without specifying `device` doesn't silently land tensors on CPU and
        # crash inside `model.input_proj` with a device-mismatch error.
        if device is not None:
            self.device: torch.device = torch.device(device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                # No parameters (extremely unusual; only happens if someone
                # constructs a stripped TurnRD). Fall back to CPU.
                self.device = torch.device("cpu")
        # Resolve the model's parameter dtype too: the embedder is allowed to
        # return any dtype, and we'll cast `stacked` into the model dtype
        # before forward. This avoids the fp16-embedder vs fp32-input_proj
        # dtype-mismatch RuntimeError that would otherwise greet the Day-14
        # production wiring.
        try:
            self._model_dtype: torch.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self._model_dtype = torch.float32

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        """Return list[K] of list[T_i] per-turn rewards `r̂_t = α_t · R`.

        Steps:
        1. Embed each non-empty trajectory via `self.embedder` → `[T_i, D]`,
           inside `torch.no_grad()` so a careless embedder can't leak the
           policy's autograd graph into our K×T padded stack.
        2. Pad to `[K, T_max, D]` (cast to the model's param device+dtype)
           and build `attention_mask = [K, T_max]`.
        3. Forward through `model` under `torch.no_grad()` + `eval()`.
        4. Multiply attention weights by per-trajectory `final_reward`.
        5. Slice each row back to its real T_i and convert to Python floats.
           Empty trajectories return `[]` and the model is NOT called for
           them (matches `JudgeDecomposer.decompose` behavior).
        """
        K = len(group.trajectories)
        if K == 0:
            return []

        # 1. Embed each non-empty trajectory under `no_grad` so a careless
        #    embedder (one that forgot its own `with torch.no_grad():` /
        #    `.detach()`) doesn't keep the LoRA-policy backward graph alive
        #    across rollout groups.
        per_traj_embeds: list[Optional[torch.Tensor]] = []
        with torch.no_grad():
            for traj in group.trajectories:
                if not traj.turns:
                    per_traj_embeds.append(None)
                    continue
                embed = self.embedder(traj)
                if embed.dim() != 2:
                    raise ValueError(
                        f"embedder(traj) must return [T_i, D]; got shape "
                        f"{tuple(embed.shape)} for task_id={traj.task_id}."
                    )
                if embed.shape[0] != len(traj.turns):
                    raise ValueError(
                        f"embedder returned T={embed.shape[0]} but trajectory has "
                        f"{len(traj.turns)} turns (task_id={traj.task_id})."
                    )
                # Defense in depth: detach in case the embedder built its
                # tensor BEFORE the `no_grad` context took effect (e.g. it
                # captured a closure over a grad-enabled tensor).
                per_traj_embeds.append(embed.detach())

        # If all trajectories are empty, short-circuit before allocating
        # an empty-T tensor (the model would reject T=0).
        non_empty = [e for e in per_traj_embeds if e is not None]
        if not non_empty:
            return [[] for _ in range(K)]

        # 2. Pad and stack across the non-empty trajectories.
        D = non_empty[0].shape[1]
        for e in non_empty[1:]:
            if e.shape[1] != D:
                raise ValueError(
                    "embedder returned inconsistent D across trajectories: "
                    f"first={D}, this={e.shape[1]}."
                )
        T_max = max(e.shape[0] for e in non_empty)

        # Build padded stack ONLY over non-empty trajectories. We'll splice
        # the empties back in at the end so we don't waste a model call on
        # an all-zero row.
        nonempty_indices = [i for i, e in enumerate(per_traj_embeds) if e is not None]
        K_real = len(nonempty_indices)

        # Cast to the model's param device + dtype (resolved once in __init__).
        # The embedder is allowed to return any device/dtype combo per the
        # documented contract; the adapter normalises here so the model's
        # `input_proj` never trips on a device-mismatch / dtype-mismatch error.
        target_device = self.device
        target_dtype = self._model_dtype

        stacked = torch.zeros(K_real, T_max, D, dtype=target_dtype, device=target_device)
        attn_mask = torch.zeros(K_real, T_max, dtype=torch.long, device=target_device)
        for row, traj_idx in enumerate(nonempty_indices):
            e = per_traj_embeds[traj_idx]
            assert e is not None  # narrowed for the type checker
            T_i = e.shape[0]
            stacked[row, :T_i] = e.to(device=target_device, dtype=target_dtype)
            attn_mask[row, :T_i] = 1

        # 3. Forward in eval mode with no grad. Use `__call__` (not `.forward`)
        #    so any `nn.Module` forward-pre/post hooks the trainer attaches
        #    on Day 14 (e.g. refresh-cadence telemetry) still fire.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model(stacked, attn_mask)
        finally:
            if was_training:
                self.model.train()

        # 4. Build final_R aligned with the non-empty rows.
        final_R = torch.tensor(
            [float(group.trajectories[i].final_reward) for i in nonempty_indices],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        per_turn = out.decompose(final_R)  # [K_real, T_max]
        # Defensive: zero padded slots so any tiny float drift doesn't sneak in.
        per_turn = per_turn * attn_mask.to(dtype=per_turn.dtype)

        # 5. Splice back, returning [] for empty trajectories.
        per_turn_cpu = per_turn.cpu()
        out_list: list[list[float]] = []
        cursor = 0
        for i, traj in enumerate(group.trajectories):
            if not traj.turns:
                out_list.append([])
                continue
            T_i = len(traj.turns)
            row = per_turn_cpu[cursor, :T_i].tolist()
            out_list.append([float(x) for x in row])
            cursor += 1
        return out_list

    # -------------------------------------------------------------------
    # Learnable surface (Day 13)
    # -------------------------------------------------------------------

    def __call__(self, group: TrajectoryGroup) -> list[list[float]]:
        """Make the decomposer object directly callable.

        The trainer's `build_advantages` invokes `self.decomposer(group)`
        for stats reporting (Methods A/C path); making the object
        callable lets the user pass a `TurnRDDecomposer` instance as the
        trainer's `decomposer` argument and still reach
        `decompose_with_grad` / `parameters` / `has_learnable_params`
        on the same object.
        """
        return self.decompose(group)

    @property
    def has_learnable_params(self) -> bool:
        """TurnRD is the learnable Method B decomposer.

        The trainer (`HGPOTrainer.__init__` + `compute_loss`) reads this
        via `getattr(decomposer, "has_learnable_params", False)` so other
        decomposers (Methods A/C) need NOT define it — they implicitly
        read False and the trainer skips the second optimizer + the C3
        consistency-loss reattach for them.
        """
        return True

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Forward to `self.model.parameters()` so the trainer can build a
        separate AdamW for the TurnRD params (`turnrd_lr` in the config)."""
        return self.model.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Forward to `self.model.state_dict()` for the refresh hook +
        Modal checkpoint persistence."""
        return self.model.state_dict()

    def load_state_dict(
        self,
        sd: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> Any:
        """Forward to `self.model.load_state_dict(...)`. Returned value is
        whatever PyTorch returns (an `IncompatibleKeys` namedtuple); kept
        as `Any` so we don't add a torch internals dep here."""
        return self.model.load_state_dict(sd, strict=strict)

    def decompose_with_grad(self, group: TrajectoryGroup) -> dict[str, Any]:
        """Grad-enabled twin of `decompose`, used by `HGPOTrainer.compute_loss`
        to build the C3 consistency loss against TurnRD params.

        Differences from `decompose`:
        - Does NOT enter `torch.no_grad()` around the model forward (we
          want grad to flow back to TurnRD params).
        - Does NOT call `model.eval()` (TurnRD is *training* on this path).
        - Returns a dict of tensors instead of a Python list, so the
          caller can compute torch-tensor advantages without re-padding:
            * `alpha`:           `[K_real, T_max]` (the model's α weights)
            * `attention_mask`:  `[K_real, T_max]` long
            * `nonempty_indices`: `list[int]` (indices into `group.trajectories`
                                  for the rows present in `alpha`)
            * `final_R`:         `[K_real]` aligned with `alpha` rows.

        Important: the embedder loop still runs under `torch.no_grad()`
        and detaches each returned tensor — the gradient path we care
        about is α_t → TurnRD params (NOT into the policy backbone the
        embedder hits).
        """
        K = len(group.trajectories)
        if K == 0:
            return {
                "alpha": torch.zeros(0, 0, device=self.device, dtype=self._model_dtype),
                "attention_mask": torch.zeros(0, 0, dtype=torch.long, device=self.device),
                "nonempty_indices": [],
                "final_R": torch.zeros(0, device=self.device, dtype=self._model_dtype),
                "value_per_turn": None,
            }

        # 1. Embed each non-empty trajectory under no_grad (same rationale
        #    as `decompose`: we only want gradient through TurnRD params).
        per_traj_embeds: list[Optional[torch.Tensor]] = []
        with torch.no_grad():
            for traj in group.trajectories:
                if not traj.turns:
                    per_traj_embeds.append(None)
                    continue
                embed = self.embedder(traj)
                if embed.dim() != 2:
                    raise ValueError(
                        f"embedder(traj) must return [T_i, D]; got shape "
                        f"{tuple(embed.shape)} for task_id={traj.task_id}."
                    )
                if embed.shape[0] != len(traj.turns):
                    raise ValueError(
                        f"embedder returned T={embed.shape[0]} but trajectory has "
                        f"{len(traj.turns)} turns (task_id={traj.task_id})."
                    )
                per_traj_embeds.append(embed.detach())

        non_empty = [e for e in per_traj_embeds if e is not None]
        if not non_empty:
            return {
                "alpha": torch.zeros(0, 0, device=self.device, dtype=self._model_dtype),
                "attention_mask": torch.zeros(0, 0, dtype=torch.long, device=self.device),
                "nonempty_indices": [],
                "final_R": torch.zeros(0, device=self.device, dtype=self._model_dtype),
                "value_per_turn": None,
            }

        D = non_empty[0].shape[1]
        for e in non_empty[1:]:
            if e.shape[1] != D:
                raise ValueError(
                    "embedder returned inconsistent D across trajectories: "
                    f"first={D}, this={e.shape[1]}."
                )
        T_max = max(e.shape[0] for e in non_empty)
        nonempty_indices = [i for i, e in enumerate(per_traj_embeds) if e is not None]
        K_real = len(nonempty_indices)

        target_device = self.device
        target_dtype = self._model_dtype

        stacked = torch.zeros(K_real, T_max, D, dtype=target_dtype, device=target_device)
        attn_mask = torch.zeros(K_real, T_max, dtype=torch.long, device=target_device)
        for row, traj_idx in enumerate(nonempty_indices):
            e = per_traj_embeds[traj_idx]
            assert e is not None
            T_i = e.shape[0]
            stacked[row, :T_i] = e.to(device=target_device, dtype=target_dtype)
            attn_mask[row, :T_i] = 1

        # NOTE: no torch.no_grad(), no eval() — we WANT grad through the
        # model. The trainer is responsible for putting the model in
        # train() mode before calling this.
        out = self.model(stacked, attn_mask)
        # alpha == cls_attn_weights (already mask-zeroed inside the model).
        alpha = out.cls_attn_weights  # [K_real, T_max], grad-tracking
        # v6: V-head per-turn predictions for the actor-critic baseline
        # in HGPOTrainer.compute_loss. None when the model was built
        # with cfg.value_head=False (back-compat); the trainer falls
        # back to the per-position-normalized turn_adv path then.
        value_per_turn = out.predicted_per_turn_R  # [K_real, T_max] or None

        final_R = torch.tensor(
            [float(group.trajectories[i].final_reward) for i in nonempty_indices],
            dtype=stacked.dtype,
            device=stacked.device,
        )

        return {
            "alpha": alpha,
            "attention_mask": attn_mask,
            "nonempty_indices": nonempty_indices,
            "final_R": final_R,
            "value_per_turn": value_per_turn,
        }


def build_turnrd_decomposer(
    cfg: dict[str, Any],
    *,
    model: TurnRD,
    embedder: TurnEmbedder,
    device: Optional[str] = None,
) -> "TurnRDDecomposer":
    """Factory used by `build_decomposer` for the `"turnrd"` branch.

    Returns the `TurnRDDecomposer` *object* (not its `.decompose` method)
    so the trainer can reach the Day-13 learnable surface
    (`has_learnable_params`, `parameters`, `decompose_with_grad`,
    `state_dict`, `load_state_dict`). `TurnRDDecomposer.__call__` forwards
    to `.decompose`, so the returned value is still a valid
    `PerTurnDecomposer` per the existing call-site contract
    `self.decomposer(group)` in `HGPOTrainer.build_advantages`.
    """
    # cfg is accepted for symmetry with the other build_* factories; future
    # config-loader work will read e.g. cfg["turnrd"]["refresh_every_n_episodes"].
    _ = cfg
    return TurnRDDecomposer(model=model, embedder=embedder, device=device)
