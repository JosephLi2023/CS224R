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

from typing import Any, Callable, Optional

import torch

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup
from src.turnrd.model import TurnRD


# Embedder contract: per-trajectory callable returning a per-turn embedding
# tensor of shape `[T_i, D]` (D == TurnRD.input_dim). Production wires this
# from `LoRAPolicy.model.eval()` mean-pooled hidden states (Day 14); tests
# pass a deterministic stub.
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
        self.device = device

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        """Return list[K] of list[T_i] per-turn rewards `r̂_t = α_t · R`.

        Steps:
        1. Embed each non-empty trajectory via `self.embedder` → `[T_i, D]`.
        2. Pad to `[K, T_max, D]` and build `attention_mask = [K, T_max]`.
        3. Forward through `model` under `torch.no_grad()` + `eval()`.
        4. Multiply attention weights by per-trajectory `final_reward`.
        5. Slice each row back to its real T_i and convert to Python floats.
           Empty trajectories return `[]` and the model is NOT called for
           them (matches `JudgeDecomposer.decompose` behavior).
        """
        K = len(group.trajectories)
        if K == 0:
            return []

        # 1. Embed each non-empty trajectory.
        per_traj_embeds: list[Optional[torch.Tensor]] = []
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
            per_traj_embeds.append(embed)

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

        ref_dtype = non_empty[0].dtype
        ref_device = non_empty[0].device
        target_device = (
            torch.device(self.device) if self.device is not None else ref_device
        )

        stacked = torch.zeros(K_real, T_max, D, dtype=ref_dtype, device=target_device)
        attn_mask = torch.zeros(K_real, T_max, dtype=torch.long, device=target_device)
        for row, traj_idx in enumerate(nonempty_indices):
            e = per_traj_embeds[traj_idx]
            assert e is not None  # narrowed for the type checker
            T_i = e.shape[0]
            stacked[row, :T_i] = e.to(device=target_device, dtype=ref_dtype)
            attn_mask[row, :T_i] = 1

        # 3. Forward in eval mode with no grad.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                out = self.model.forward(stacked, attn_mask)
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


def build_turnrd_decomposer(
    cfg: dict[str, Any],
    *,
    model: TurnRD,
    embedder: TurnEmbedder,
    device: Optional[str] = None,
) -> Callable[[TrajectoryGroup], list[list[float]]]:
    """Factory used by `build_decomposer` for the `"turnrd"` branch.

    Reads no extra config today; placeholder for future Day 14 hooks
    (e.g. refresh cadence wiring). Returns the decomposer's `decompose`
    method so it satisfies the `PerTurnDecomposer` callable type.
    """
    # cfg is accepted for symmetry with the other build_* factories; future
    # Day 14 work will read e.g. cfg["turnrd"]["refresh_every_n_episodes"].
    _ = cfg
    decomposer = TurnRDDecomposer(model=model, embedder=embedder, device=device)
    return decomposer.decompose
