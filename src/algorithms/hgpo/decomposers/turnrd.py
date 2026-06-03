"""TurnRD per-turn reward decomposer adapter (Method B).

Inference-only adapter that the trainer plugs in. The embedder callable is
supplied by the caller, so tests can drive it with a deterministic stub. The
invariant `sum_t r_hat_t = R` holds by construction (`decompose(R) = alpha * R`
with `sum alpha = 1` after masking).
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional, Union

import torch

from src.algorithms.grpo.rollout import Trajectory, TrajectoryGroup, TurnRecord
from src.turnrd.model import TurnRD, TurnRDv2


def _make_goal_only_turn(goal_text: str) -> TurnRecord:
    """Build a single-turn `TurnRecord` whose observation_text is the goal
    text, used to drive the per-trajectory goal embedder.
    """
    return TurnRecord(
        turn_idx=0,
        observation_text=goal_text,
        action_text="",
        raw_env_reward=0.0,
    )

# Type alias for any TurnRD-shaped model. Both `TurnRD` (v1) and `TurnRDv2`
# expose the same `TurnRDOutput`, so the adapter is architecture-agnostic.
TurnRDLike = Union[TurnRD, TurnRDv2]


# Embedder contract: per-trajectory callable returning a per-turn embedding
# of shape `[T_i, D]` (D == TurnRD.input_dim). Device/dtype are free (the
# adapter casts to the model's before forward); the returned tensor should be
# detached, though the adapter also wraps the call in `torch.no_grad()`.
TurnEmbedder = Callable[[Trajectory], torch.Tensor]


class TurnRDDecomposer:
    """Per-turn reward decomposer that delegates to a learned `TurnRD` model.

    Plugs into `HGPOTrainer(decomposer=TurnRDDecomposer(...).decompose)`.
    """

    def __init__(
        self,
        model: TurnRDLike,
        embedder: TurnEmbedder,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.embedder = embedder
        # Resolve the target device once, defaulting to the model's parameter
        # device so a CPU embedder + CUDA model don't trip a device mismatch.
        if device is not None:
            self.device: torch.device = torch.device(device)
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                # No parameters (stripped TurnRD); fall back to CPU.
                self.device = torch.device("cpu")
        # Resolve the model's param dtype; `stacked` is cast to it before
        # forward to avoid an embedder-vs-input_proj dtype mismatch.
        try:
            self._model_dtype: torch.dtype = next(self.model.parameters()).dtype
        except StopIteration:
            self._model_dtype = torch.float32
        # When goal conditioning is enabled, compute one goal embedding per
        # trajectory; the cache is local to one decompose call.
        self._goal_emb_enabled: bool = bool(
            getattr(getattr(self.model, "cfg", None), "goal_conditioned_value_head", False)
        )
        # Stash the most recent eval-mode attention + alignment metadata so the
        # trainer can compute alpha diagnostics off the no-grad path:
        #   _last_alpha / _last_alpha_mask: CPU [K_real, T_max] (detached, 0/1)
        #   _last_alpha_traj_indices: K_real rows -> original K-index
        self._last_alpha: Optional[torch.Tensor] = None
        self._last_alpha_mask: Optional[torch.Tensor] = None
        self._last_alpha_traj_indices: list[int] = []

    def _compute_goal_emb_for_indices(
        self,
        group: "TrajectoryGroup",
        nonempty_indices: list[int],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute `(goal_emb [K_real, D], goal_emb_mask [K_real])` for the
        FiLM goal-conditioned V-head.

        Returns `(None, None)` when goal conditioning is disabled. Otherwise
        extracts each trajectory's goal text, embeds it via `self.embedder`, and
        stacks per-row; rows whose goal doesn't parse get mask=0 and fall back
        to the unconditioned path.
        """
        if not self._goal_emb_enabled or not nonempty_indices:
            return (None, None)
        # Lazy import keeps the module torch-only.
        try:
            from src.turnrd.goal_extractor import extract_goal_text  # type: ignore[import-not-found]
        except Exception:
            return (None, None)

        target_device = self.device
        target_dtype = self._model_dtype
        K_real = len(nonempty_indices)
        D = int(self.model.input_dim)
        goal_emb = torch.zeros(K_real, D, dtype=target_dtype, device=target_device)
        goal_emb_mask = torch.zeros(K_real, dtype=target_dtype, device=target_device)
        # Cache by goal_text within this call so K rollouts of the same task
        # share one embedder forward pass.
        cache: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for row, traj_idx in enumerate(nonempty_indices):
                traj = group.trajectories[traj_idx]
                if not traj.turns:
                    continue
                obs_text = traj.turns[0].observation_text or ""
                goal_text = extract_goal_text(obs_text)
                if not goal_text:
                    continue
                cached = cache.get(goal_text)
                if cached is not None:
                    goal_emb[row] = cached
                    goal_emb_mask[row] = 1.0
                    continue
                try:
                    synth = Trajectory(
                        task_id=str(traj.task_id),
                        env_name=traj.env_name,
                        turns=[
                            _make_goal_only_turn(goal_text)
                        ],
                        final_reward=0.0,
                    )
                    ge_t = self.embedder(synth)
                    if ge_t.dim() != 2 or ge_t.shape[0] < 1 or ge_t.shape[1] != D:
                        # Wrong shape: skip this row (mask stays 0); don't
                        # raise so an embedder bug doesn't kill the train step.
                        continue
                    row_t = ge_t[0].detach().to(device=target_device, dtype=target_dtype)
                    goal_emb[row] = row_t
                    goal_emb_mask[row] = 1.0
                    cache[goal_text] = row_t
                except Exception:
                    # Defensive: a single embedder failure shouldn't propagate.
                    continue
        return (goal_emb, goal_emb_mask)

    def decompose(self, group: TrajectoryGroup) -> list[list[float]]:
        """Return list[K] of list[T_i] per-turn rewards `r_hat_t = alpha_t * R`.

        Embeds each non-empty trajectory under `torch.no_grad()`, pads to
        `[K, T_max, D]`, forwards through the model in eval mode, and scales the
        attention weights by `final_reward`. Empty trajectories return `[]` and
        are not forwarded.
        """
        K = len(group.trajectories)
        if K == 0:
            return []

        # 1. Embed each non-empty trajectory under `no_grad` so a careless
        #    embedder can't keep the policy backward graph alive across groups.
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
                # Detach in case the embedder built its tensor before the
                # `no_grad` context took effect.
                per_traj_embeds.append(embed.detach())

        # If all trajectories are empty, short-circuit (the model rejects T=0).
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

        # Build the padded stack over non-empty trajectories only; empties are
        # spliced back at the end to avoid a wasted all-zero model row.
        nonempty_indices = [i for i, e in enumerate(per_traj_embeds) if e is not None]
        K_real = len(nonempty_indices)

        # Cast to the model's param device + dtype so `input_proj` never trips
        # on a device/dtype mismatch.
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

        # 3. Forward in eval mode with no grad, via `__call__` so any module
        #    hooks still fire. Compute goal_emb only when FiLM conditioning is on.
        goal_emb, goal_emb_mask = self._compute_goal_emb_for_indices(
            group, nonempty_indices
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                if goal_emb is not None:
                    out = self.model(
                        stacked, attn_mask,
                        goal_emb=goal_emb,
                        goal_emb_mask=goal_emb_mask,
                    )
                else:
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
        # Zero padded slots so float drift doesn't sneak in.
        per_turn = per_turn * attn_mask.to(dtype=per_turn.dtype)

        # Expose eval-mode attention (detached, CPU) for trainer diagnostics.
        self._last_alpha = out.cls_attn_weights.detach().to(
            dtype=torch.float32, device=torch.device("cpu")
        )
        self._last_alpha_mask = attn_mask.detach().to(
            dtype=torch.float32, device=torch.device("cpu")
        )
        self._last_alpha_traj_indices = list(nonempty_indices)

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

    # Learnable surface

    def __call__(self, group: TrajectoryGroup) -> list[list[float]]:
        """Forward to `decompose` so a `TurnRDDecomposer` instance can be passed
        as the trainer's `decomposer` while exposing the learnable surface.
        """
        return self.decompose(group)

    @property
    def has_learnable_params(self) -> bool:
        """True: TurnRD is the learnable Method B decomposer. The trainer reads
        this via `getattr` to enable the second optimizer + C3 consistency loss.
        """
        return True

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Forward to `self.model.parameters()` for the trainer's TurnRD AdamW."""
        return self.model.parameters()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Forward to `self.model.state_dict()` for refresh + checkpointing."""
        return self.model.state_dict()

    def load_state_dict(
        self,
        sd: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> Any:
        """Forward to `self.model.load_state_dict(...)`; returns PyTorch's
        `IncompatibleKeys`, typed `Any` to avoid a torch-internals dep."""
        return self.model.load_state_dict(sd, strict=strict)

    def decompose_with_grad(self, group: TrajectoryGroup) -> dict[str, Any]:
        """Grad-enabled twin of `decompose` for `HGPOTrainer.compute_loss`'s C3
        consistency loss: skips `no_grad`/`eval` around the forward so grad flows
        to TurnRD params (the embedder loop still runs under `no_grad`). Returns
        a dict: `alpha` [K_real, T_max], `attention_mask`, `nonempty_indices`,
        `final_R` [K_real], and `value_per_turn`.
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

        # 1. Embed under no_grad (same rationale as `decompose`).
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

        # Keep gradients through the decomposer (trainer controls train/eval).
        # FiLM uses the same goal-embedding path; grad flows through goal_proj/
        # goal_gamma/goal_beta but not the embedder.
        goal_emb, goal_emb_mask = self._compute_goal_emb_for_indices(
            group, nonempty_indices
        )
        if goal_emb is not None:
            out = self.model(
                stacked, attn_mask,
                goal_emb=goal_emb,
                goal_emb_mask=goal_emb_mask,
            )
        else:
            out = self.model(stacked, attn_mask)
        # alpha == cls_attn_weights (already mask-zeroed inside the model).
        alpha = out.cls_attn_weights  # [K_real, T_max], grad-tracking
        # V-head per-turn predictions for the actor-critic baseline; None when
        # the model was built without a value head (trainer falls back).
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
    model: TurnRDLike,
    embedder: TurnEmbedder,
    device: Optional[str] = None,
) -> "TurnRDDecomposer":
    """Factory for `build_decomposer`'s `"turnrd"` branch.

    Returns the `TurnRDDecomposer` object (not its `.decompose`) so the trainer
    can reach the learnable surface; `__call__` forwards to `.decompose`, so it
    still satisfies the `PerTurnDecomposer` contract.
    """
    # cfg accepted for symmetry with the other build_* factories.
    _ = cfg
    return TurnRDDecomposer(model=model, embedder=embedder, device=device)
