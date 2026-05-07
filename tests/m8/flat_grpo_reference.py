"""Independent flat-GRPO loss reference implementation (M8 verification gate).

This module deliberately does NOT import from `src.algorithms.grpo.*`. It is a
fresh re-derivation used as an independent oracle in
`tests/m8/test_flat_grpo_equivalence.py` to catch formula regressions in the
trainer's pure-Python helpers.

The reference covers the FLAT-GRPO setting (α=1, λ=0, kl_coef=0). For α<1 we
verify component-wise (advantage construction) elsewhere.
"""

from __future__ import annotations

import math


_SIGMA_FLOOR = 1e-8


def reference_traj_advantages(final_rewards: list[float]) -> list[float]:
    """Group-normalised trajectory advantage: Â_i = (R_i − R̄) / σ_R.

    Uses *population* standard deviation (divide by K, not K−1) to match the
    trainer; floors σ at SIGMA_FLOOR to avoid divide-by-zero on degenerate
    groups.
    """
    K = len(final_rewards)
    if K == 0:
        return []
    R_mean = sum(final_rewards) / K
    var = sum((r - R_mean) ** 2 for r in final_rewards) / K
    sigma = math.sqrt(var + _SIGMA_FLOOR ** 2)
    return [(r - R_mean) / sigma for r in final_rewards]


def reference_flat_grpo_loss(
    *,
    final_rewards: list[float],
    per_trajectory_new_logprobs: list[list[float]],
    per_trajectory_old_logprobs: list[list[float]],
    clip_eps: float,
) -> float:
    """End-to-end flat-GRPO loss: -mean( min(ρ·Â, clip(ρ,1±ε)·Â) ).

    Flat-GRPO setting (α=1, λ=0, kl_coef=0):
      1. Â_i = (R_i − R̄)/σ_R
      2. broadcast Â_i to every token in trajectory i
      3. ρ_t = exp(new_lp_t − old_lp_t)
      4. L_t = -min(ρ_t·Â_i, clamp(ρ_t, 1-ε, 1+ε)·Â_i)
      5. mean over all real action tokens
    """
    K = len(final_rewards)
    if K == 0:
        return 0.0
    if (
        len(per_trajectory_new_logprobs) != K
        or len(per_trajectory_old_logprobs) != K
    ):
        raise ValueError("logprob lists must have one entry per trajectory")

    advantages = reference_traj_advantages(final_rewards)
    lo, hi = 1.0 - clip_eps, 1.0 + clip_eps

    losses: list[float] = []
    for i in range(K):
        adv_i = advantages[i]
        new_lps = per_trajectory_new_logprobs[i]
        old_lps = per_trajectory_old_logprobs[i]
        if len(new_lps) != len(old_lps):
            raise ValueError(
                f"trajectory {i}: new_lps len {len(new_lps)} != old_lps {len(old_lps)}"
            )
        for new_lp, old_lp in zip(new_lps, old_lps):
            rho = math.exp(new_lp - old_lp)
            unclipped = rho * adv_i
            clipped = max(lo, min(hi, rho)) * adv_i
            losses.append(-min(unclipped, clipped))

    if not losses:
        return 0.0
    return sum(losses) / len(losses)
