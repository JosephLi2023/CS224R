"""PPO loss building blocks for the H-GRPO trainer.

Pure-Python operating on `list[float]` so the math is unit-testable without a
GPU. The actual trainer (`src.algorithms.grpo.trainer.HGPOTrainer`) calls the
torch-native equivalents inline (`torch.exp`, `torch.clamp`, etc.) so these
helpers serve as a numerical reference that the torch path is checked
against in `tests/unit/test_grpo_loss.py`.
"""
from __future__ import annotations

import math


def importance_ratio(new_logprobs: list[float], old_logprobs: list[float]) -> list[float]:
    """rho_t = exp(new_logprob_t - old_logprob_t)."""
    if len(new_logprobs) != len(old_logprobs):
        raise ValueError(
            f"length mismatch: new={len(new_logprobs)} old={len(old_logprobs)}"
        )
    return [math.exp(n - o) for n, o in zip(new_logprobs, old_logprobs)]


def clipped_ppo_term(
    ratios: list[float],
    advantages: list[float],
    clip_eps: float,
) -> list[float]:
    """Schulman PPO clipped surrogate (per-token):

        L_t = - min( rho_t * A_t , clamp(rho_t, 1 - eps, 1 + eps) * A_t )

    Returns a per-token list (sign already flipped so it's a *loss*, mean-reducing
    over real tokens gives the policy loss). Caller masks padding tokens
    via `mask_mean(...)`.
    """
    if len(ratios) != len(advantages):
        raise ValueError(
            f"length mismatch: ratios={len(ratios)} advantages={len(advantages)}"
        )
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative")

    out: list[float] = []
    lo, hi = 1.0 - clip_eps, 1.0 + clip_eps
    for r, a in zip(ratios, advantages):
        unclipped = r * a
        clipped = max(lo, min(hi, r)) * a
        # `min(unclipped, clipped)` is the conservative reward; loss is its negation.
        out.append(-min(unclipped, clipped))
    return out


def mask_mean(values: list[float], mask: list[float | int]) -> float:
    """Mean of `values` over positions where `mask > 0` (binary semantics:
    each active position contributes equal weight). Returns 0 when no
    positions are active (instead of NaN)."""
    if len(values) != len(mask):
        raise ValueError(
            f"length mismatch: values={len(values)} mask={len(mask)}"
        )
    s = 0.0
    n = 0
    for v, m in zip(values, mask):
        if m > 0:
            s += v
            n += 1
    if n == 0:
        return 0.0
    return s / n


def kl_per_token(
    new_logprobs: list[float], old_logprobs: list[float]
) -> list[float]:
    """Per-token KL approximation `new_logprob - old_logprob` (Schulman's k1
    estimator). This is unbiased but high-variance; trainer can switch to k3
    `(ratio - 1) - log(ratio)` for lower variance if needed."""
    if len(new_logprobs) != len(old_logprobs):
        raise ValueError(
            f"length mismatch: new={len(new_logprobs)} old={len(old_logprobs)}"
        )
    return [n - o for n, o in zip(new_logprobs, old_logprobs)]


def kl_k3_per_token(
    new_logprobs: list[float], old_logprobs: list[float]
) -> list[float]:
    """Per-token KL k3 estimator: (rho - 1) - log(rho), where rho = exp(new - old).
    Lower variance than k1 but still unbiased."""
    out: list[float] = []
    for n, o in zip(new_logprobs, old_logprobs):
        diff = n - o
        rho = math.exp(diff)
        out.append((rho - 1.0) - diff)
    return out
