"""Residual per-turn reward decomposer (Method D).

Hybrid of Method C (Progress) and Method B (TurnRD): the per-turn
`raw_env_reward` from the env is used as a *prior bias* on the α
softmax, and TurnRD learns only the *residual* correction. A
learnable scalar `gamma_prior` controls how much the model trusts
the prior:

    α_t = softmax_t( transformer_logits_t  +  gamma_prior · raw_env_reward_t )
    r̂_t = α_t · R          (sum-to-R invariant preserved)

- `gamma_prior → 0` reduces Method D to Method B (TurnRD).
- `gamma_prior → ∞` collapses Method D toward Method C (progress weights).
- Learnable `gamma_prior` (`nn.Parameter`, init = 1.0) lets the trainer's
  AdamW choose the operating point during training.

This file mirrors `src.algorithms.hgpo.decomposers.turnrd` for the
inference adapter contract; the difference is in `decompose` and
`decompose_with_grad`, which build a `[K_real, T_max]` prior tensor
from `traj.turns[t].raw_env_reward` and pass `prior_bias =
gamma_prior * prior` to `TurnRD.forward`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn as nn

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup
from src.turnrd.model import TurnRD


# Same embedder contract as TurnRDDecomposer.
TurnEmbedder = Callable[[Trajectory], torch.Tensor]


class ResidualDecomposer:
    """Per-turn reward decomposer that adds a `gamma_prior · raw_env_reward`
    bias to TurnRD's α softmax (Method D).

    Plugs into `HGPOTrainer(decomposer=ResidualDecomposer(...))`. The
    surface mirrors `TurnRDDecomposer` so the trainer's
    `getattr(decomposer, "has_learnable_params", False)` and
    `decompose_with_grad` paths work without modification.
    """

    def __init__(
        self,
        model: TurnRD,
        embedder: TurnEmbedder,
        device: Optional[str] = None,
        *,
        init_gamma: float = 1.0,
    ) -> None:
        self.model = model
        self.embedder = embedder
        if device is not None:
            self.device: torch.device = torch.device(device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        try:
            self._model_dtype: torch.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self._model_dtype = torch.float32

        # Learnable scalar that scales the per-turn raw_env_reward prior
        # before it is added to the cls-pool logits. Lives on the same
        # device + dtype as the TurnRD model so the trainer's AdamW
        # finds it via `parameters()` and updates it under the same
        # `turnrd_lr`. Stored as `nn.Parameter` (not a buffer) so it
        # appears in `state_dict()` AND receives gradients.
        self.gamma_prior = nn.Parameter(
            torch.tensor(float(init_gamma), dtype=self._model_dtype, device=self.device)
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _build_prior(
        self,
        group: TrajectoryGroup,
        nonempty_indices: list[int],
        T_max: int,
    ) -> torch.Tensor:
        """Build `[K_real, T_max]` per-turn raw_env_reward tensor (padded with 0).

        Padded positions are 0.0 — a 0 bias is a no-op inside softmax,
        and the post-pool mask renormalization in `TurnRD.forward`
        zeros padded α anyway, so the choice is correct + safe.
        """
        K_real = len(nonempty_indices)
        prior = torch.zeros(
            K_real, T_max, dtype=self._model_dtype, device=self.device
        )
        for row, traj_idx in enumerate(nonempty_indices):
            traj = group.trajectories[traj_idx]
            for t, turn in enumerate(traj.turns):
                prior[row, t] = float(turn.raw_env_reward)
        return prior

    # -------------------------------------------------------------------
    # Inference path
    # -------------------------------------------------------------------

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        """Return list[K] of list[T_i] per-turn rewards `r̂_t = α_t · R`.

        Same structure as `TurnRDDecomposer.decompose`, but builds and
        passes `prior_bias = gamma_prior * raw_env_reward` to the
        model so the α softmax is biased toward env-reported progress.
        Empty trajectories return `[]` and the model is NOT called.
        """
        K = len(group.trajectories)
        if K == 0:
            return []

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
            return [[] for _ in range(K)]

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

        prior = self._build_prior(group, nonempty_indices, T_max)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                # Use detached gamma_prior on the inference path — no need
                # to track grad through stats reporting.
                bias = self.gamma_prior.detach() * prior
                out = self.model(stacked, attn_mask, prior_bias=bias)
        finally:
            if was_training:
                self.model.train()

        final_R = torch.tensor(
            [float(group.trajectories[i].final_reward) for i in nonempty_indices],
            dtype=stacked.dtype,
            device=stacked.device,
        )
        per_turn = out.decompose(final_R)
        per_turn = per_turn * attn_mask.to(dtype=per_turn.dtype)

        per_turn_cpu = per_turn.cpu()
        out_list: list[list[float]] = []
        cursor = 0
        for traj in group.trajectories:
            if not traj.turns:
                out_list.append([])
                continue
            T_i = len(traj.turns)
            row = per_turn_cpu[cursor, :T_i].tolist()
            out_list.append([float(x) for x in row])
            cursor += 1
        return out_list

    # -------------------------------------------------------------------
    # Learnable surface (mirrors TurnRDDecomposer)
    # -------------------------------------------------------------------

    def __call__(self, group: TrajectoryGroup) -> list[list[float]]:
        return self.decompose(group)

    @property
    def has_learnable_params(self) -> bool:
        return True

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Yield TurnRD model params + the residual `gamma_prior` so the
        trainer's AdamW updates both with the same `turnrd_lr`."""
        yield from self.model.parameters()
        yield self.gamma_prior

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Bundle TurnRD's state_dict with `gamma_prior` so checkpointing
        round-trips both. Use a `gamma_prior` key prefix to avoid clashes
        with any future TurnRD param of the same name.
        """
        sd = dict(self.model.state_dict())
        sd["gamma_prior"] = self.gamma_prior.detach().clone()
        return sd

    def load_state_dict(
        self,
        sd: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> Any:
        """Inverse of `state_dict`. Pops `gamma_prior` (if present) and
        forwards the rest to the TurnRD model. Backward-compat: a
        legacy TurnRD-only state_dict (no `gamma_prior` key) loads
        cleanly and leaves `self.gamma_prior` at its current value.
        """
        sd_local = dict(sd)
        gamma_val = sd_local.pop("gamma_prior", None)
        if gamma_val is not None:
            with torch.no_grad():
                self.gamma_prior.copy_(
                    torch.as_tensor(
                        gamma_val,
                        dtype=self.gamma_prior.dtype,
                        device=self.gamma_prior.device,
                    )
                )
        return self.model.load_state_dict(sd_local, strict=strict)

    def decompose_with_grad(self, group: TrajectoryGroup) -> dict[str, Any]:
        """Grad-enabled twin of `decompose`. Identical to
        `TurnRDDecomposer.decompose_with_grad` except the model is
        called with `prior_bias = gamma_prior * raw_env_reward`, so
        backward through `alpha` populates `gamma_prior.grad` AND
        TurnRD-model param grads.
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

        # Build prior under no_grad — only `gamma_prior` should track grad.
        with torch.no_grad():
            prior = self._build_prior(group, nonempty_indices, T_max)
        bias = self.gamma_prior * prior  # grad flows through gamma_prior

        out = self.model(stacked, attn_mask, prior_bias=bias)
        alpha = out.cls_attn_weights
        value_per_turn = out.predicted_per_turn_R

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


def build_residual_decomposer(
    cfg: dict[str, Any],
    *,
    model: TurnRD,
    embedder: TurnEmbedder,
    device: Optional[str] = None,
) -> "ResidualDecomposer":
    """Factory used by `build_decomposer` for the `"residual"` branch.

    Reads `cfg["residual"]["init_gamma"]` (default 1.0).
    """
    residual_cfg = cfg.get("residual", {}) if isinstance(cfg, dict) else {}
    init_gamma = float(residual_cfg.get("init_gamma", 1.0)) if isinstance(residual_cfg, dict) else 1.0
    return ResidualDecomposer(
        model=model, embedder=embedder, device=device, init_gamma=init_gamma
    )
