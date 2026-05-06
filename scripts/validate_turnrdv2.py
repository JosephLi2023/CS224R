"""Validation experiment: TurnRDv2 vs TurnRD (v1) on synthetic credit assignment.

Setup
-----
We construct a "needle-in-haystack" credit-assignment task where the ground-
truth causal turn is KNOWN, so we can directly measure whether each model's
α concentrates on the right turn:

  - Each trajectory has T turns of D-dim random Gaussian embeddings.
  - One specific turn index `t_star` is the CAUSAL turn:
      R = sigmoid(w · h_{t_star})  - 0.5     # ∈ [-0.5, 0.5]
    where w is a fixed random direction. R depends ONLY on h_{t_star};
    the other turns are pure distractor noise.
  - We train each model on N trajectories with their own R-prediction loss
    (loss_mode_1 for v1, loss_v2_pred for v2), then measure on a held-out
    set whether mean α[t_star] is large (model identified the needle) and
    whether the model's predicted_R correlates with the true R.

Metrics
-------
For each model:
  1. mean_alpha_at_tstar : mean of α at the causal turn across eval set
                           (should be ≫ 1/T if the model identified the causal turn)
  2. alpha_argmax_acc    : fraction of eval rows whose argmax(α) == t_star
                           (clean classification metric)
  3. R_pearson_r         : Pearson correlation of predicted_R vs true R
                           (whether the model can predict R at all)
  4. R_mse               : MSE on held-out — same scale as the training loss

A v2 win is concretely:
  - mean_alpha_at_tstar(v2) > mean_alpha_at_tstar(v1) by a clear margin
  - alpha_argmax_acc(v2) ≫ alpha_argmax_acc(v1)
  - both model R-predictions correlate with R, but v2's α is interpretable
    while v1's is uniform-ish (because v1's R-loss is non-identifiable for α).

Run
---
    python scripts/validate_turnrdv2.py

Or specify N_TRAIN / N_EVAL / SEEDS via env vars:
    N_TRAIN=512 N_EVAL=128 SEEDS=3 python scripts/validate_turnrdv2.py
"""
from __future__ import annotations

import math
import os
import statistics
from typing import Tuple

import torch

from src.turnrd.model import (
    TurnRD,
    TurnRDConfig,
    TurnRDv2,
    TurnRDv2Config,
    loss_mode_1,
    loss_v2_pred,
)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def make_dataset(
    n: int,
    T: int,
    D: int,
    t_star: int,
    *,
    w: torch.Tensor,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (embeds [n, T, D], mask [n, T] long, R [n]).

    Embeddings are i.i.d. N(0, 1). R = sigmoid(w · h_{t_star}) - 0.5 so
    R has mean ~0 and lives in [-0.5, 0.5]. Mask is all-ones (no padding)
    so we can isolate the credit-assignment behavior from the masking
    bookkeeping.
    """
    g = torch.Generator().manual_seed(seed)
    embeds = torch.randn(n, T, D, generator=g)
    mask = torch.ones(n, T, dtype=torch.long)
    causal = embeds[:, t_star, :] @ w  # [n]
    R = torch.sigmoid(causal) - 0.5
    return embeds, mask, R


# ---------------------------------------------------------------------------
# Trainers (single-loss, identical optimizer, identical n_steps for fairness)
# ---------------------------------------------------------------------------


def _train_loop(
    model: torch.nn.Module,
    loss_fn,
    embeds: torch.Tensor,
    mask: torch.Tensor,
    R: torch.Tensor,
    *,
    batch_size: int,
    n_steps: int,
    lr: float,
) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = embeds.shape[0]
    losses: list[float] = []
    model.train()
    rng = torch.Generator().manual_seed(0)
    for _step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), generator=rng)
        b_emb = embeds[idx]
        b_mask = mask[idx]
        b_R = R[idx]
        opt.zero_grad(set_to_none=True)
        out = model(b_emb, b_mask)
        loss = loss_fn(out, b_R)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-12)
    return float((x * y).sum() / denom)


def evaluate(
    model: torch.nn.Module,
    embeds: torch.Tensor,
    mask: torch.Tensor,
    R: torch.Tensor,
    t_star: int,
) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(embeds, mask)
    alpha = out.cls_attn_weights  # [N, T]
    pred = out.predicted_R  # [N]
    return {
        "mean_alpha_at_tstar": float(alpha[:, t_star].mean()),
        "alpha_argmax_acc": float((alpha.argmax(dim=-1) == t_star).float().mean()),
        "R_pearson_r": _pearson(pred, R),
        "R_mse": float(((pred - R) ** 2).mean()),
        "alpha_per_turn": alpha.mean(dim=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_one_seed(
    seed: int,
    *,
    n_train: int,
    n_eval: int,
    T: int,
    D: int,
    t_star: int,
    n_steps: int,
    batch_size: int,
    lr: float,
) -> dict:
    torch.manual_seed(seed)
    # Fixed random direction `w` — same for train and eval so the "causal
    # rule" is identifiable. Different across outer seeds.
    w = torch.randn(D)

    embeds_tr, mask_tr, R_tr = make_dataset(
        n_train, T, D, t_star, w=w, seed=1000 + seed
    )
    embeds_ev, mask_ev, R_ev = make_dataset(
        n_eval, T, D, t_star, w=w, seed=2000 + seed
    )

    # --- v1 ---
    torch.manual_seed(seed)
    v1 = TurnRD(
        TurnRDConfig(
            n_layers=2,
            hidden_size=64,
            n_heads=4,
            max_turns=T,
            dropout=0.0,
            causal=True,        # v1 default
            value_head=False,   # disable to isolate the α-vs-R-loss question
        ),
        input_dim=D,
    )
    v1_losses = _train_loop(
        v1,
        loss_mode_1,
        embeds_tr,
        mask_tr,
        R_tr,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
    )
    v1_eval = evaluate(v1, embeds_ev, mask_ev, R_ev, t_star)
    v1_eval["final_train_loss"] = v1_losses[-1]
    v1_eval["initial_train_loss"] = v1_losses[0]

    # --- v2 ---
    torch.manual_seed(seed)
    v2 = TurnRDv2(
        TurnRDv2Config(
            n_layers=2,
            hidden_size=64,
            n_heads=4,
            max_turns=T,
            dropout=0.0,
            causal=False,                  # v2 default
            progress_prior_strength=1.0,   # v2 default
        ),
        input_dim=D,
    )
    v2_losses = _train_loop(
        v2,
        loss_v2_pred,
        embeds_tr,
        mask_tr,
        R_tr,
        batch_size=batch_size,
        n_steps=n_steps,
        lr=lr,
    )
    v2_eval = evaluate(v2, embeds_ev, mask_ev, R_ev, t_star)
    v2_eval["final_train_loss"] = v2_losses[-1]
    v2_eval["initial_train_loss"] = v2_losses[0]

    return {"v1": v1_eval, "v2": v2_eval}


def _fmt(x: float, w: int = 8) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return f"{'nan':>{w}}"
    return f"{x:>{w}.4f}"


def main() -> None:
    n_train = int(os.environ.get("N_TRAIN", 512))
    n_eval = int(os.environ.get("N_EVAL", 256))
    n_steps = int(os.environ.get("N_STEPS", 400))
    batch_size = int(os.environ.get("BATCH_SIZE", 32))
    lr = float(os.environ.get("LR", 1e-3))
    n_seeds = int(os.environ.get("SEEDS", 3))
    T = int(os.environ.get("T", 6))
    D = int(os.environ.get("D", 32))
    t_star = int(os.environ.get("T_STAR", 2))

    print(
        f"\nTurnRDv2 vs TurnRD (v1) — needle-in-haystack credit assignment\n"
        f"  n_train={n_train}, n_eval={n_eval}, n_steps={n_steps}, "
        f"batch={batch_size}, lr={lr}, seeds={n_seeds}\n"
        f"  T={T}, D={D}, t_star={t_star}  (uniform α baseline = {1.0 / T:.4f})\n"
    )

    rows = []
    for s in range(n_seeds):
        r = run_one_seed(
            s,
            n_train=n_train,
            n_eval=n_eval,
            T=T,
            D=D,
            t_star=t_star,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
        )
        rows.append(r)
        print(
            f"seed {s}: "
            f"v1 α@t* = {_fmt(r['v1']['mean_alpha_at_tstar'])}  "
            f"argmax_acc = {_fmt(r['v1']['alpha_argmax_acc'])}  "
            f"R_r = {_fmt(r['v1']['R_pearson_r'])}  |  "
            f"v2 α@t* = {_fmt(r['v2']['mean_alpha_at_tstar'])}  "
            f"argmax_acc = {_fmt(r['v2']['alpha_argmax_acc'])}  "
            f"R_r = {_fmt(r['v2']['R_pearson_r'])}"
        )

    # Aggregate
    def agg(model: str, key: str) -> tuple[float, float]:
        vals = [r[model][key] for r in rows]
        return statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0

    print("\n=== aggregate over {} seeds (mean ± std) ===".format(n_seeds))
    print(
        f"  uniform-α baseline at this T : α@t* = {1.0 / T:.4f},  "
        f"argmax_acc = {1.0 / T:.4f}"
    )
    for model in ("v1", "v2"):
        m_alpha, s_alpha = agg(model, "mean_alpha_at_tstar")
        m_acc, s_acc = agg(model, "alpha_argmax_acc")
        m_r, s_r = agg(model, "R_pearson_r")
        m_mse, s_mse = agg(model, "R_mse")
        m_loss, s_loss = agg(model, "final_train_loss")
        m_init, _ = agg(model, "initial_train_loss")
        print(
            f"  {model}: α@t* = {m_alpha:.4f} ± {s_alpha:.4f}   "
            f"argmax_acc = {m_acc:.3f} ± {s_acc:.3f}   "
            f"R_r = {m_r:.3f} ± {s_r:.3f}   "
            f"R_mse = {m_mse:.4f}   "
            f"loss: {m_init:.4f} → {m_loss:.4f}"
        )

    # Per-turn α breakdown for the last seed (just to eyeball the shape)
    print("\nPer-turn mean α (last seed):")
    for model in ("v1", "v2"):
        per_turn = rows[-1][model]["alpha_per_turn"]
        marker = " ".join("*" if t == t_star else " " for t in range(T))
        print(f"  {model}: " + " ".join(f"{a:.3f}" for a in per_turn))
        print(f"  {' ' * len(model)}  " + marker + "   (* = t_star)")

    # Sanity-check verdict
    v1_alpha = agg("v1", "mean_alpha_at_tstar")[0]
    v2_alpha = agg("v2", "mean_alpha_at_tstar")[0]
    uniform = 1.0 / T
    print("\n=== verdict ===")
    if v2_alpha > v1_alpha + 0.05 and v2_alpha > uniform * 1.5:
        print(
            f"  ✓ v2 successfully concentrates α on the causal turn "
            f"(v2 α@t* = {v2_alpha:.3f} > {uniform * 1.5:.3f} = 1.5×uniform)"
        )
    elif v2_alpha > uniform * 1.5:
        print(
            f"  ~ v2 concentrates α on the causal turn ({v2_alpha:.3f}), "
            f"but v1 is comparable ({v1_alpha:.3f}) — credit-assignment "
            "advantage not demonstrated on this task."
        )
    else:
        print(
            f"  ✗ v2 α@t* = {v2_alpha:.3f} is near the uniform baseline "
            f"({uniform:.3f}); credit assignment did not work — try "
            "more steps / larger n_train / different hparams."
        )


if __name__ == "__main__":
    main()
