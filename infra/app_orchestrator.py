"""Modal cloud-side orchestrator for the H-GRPO round-loop + SFT pipeline.

This module hosts THREE `@app.function` entrypoints that drive
multi-stage pipelines from INSIDE a Modal container (24h timeout)
instead of from a laptop-resident bash/Python loop. The pattern is the
same in each case: dispatch the heavy work to other deployed Modal apps
via `modal.Function.from_name(...)`, write per-stage / per-round "done"
sentinels to the shared Volume, and auto-resume completed stages on a
preemption-restart.

## Entrypoints

| Entrypoint                  | Pattern   | Use-case                                                       |
|-----------------------------|-----------|----------------------------------------------------------------|
| `orchestrate_rl_with_turnrd` | B         | Per-round = train_loop_{env} + train_turnrd_run                |
| `orchestrate_rl_no_turnrd`   | C         | Per-round = train_loop_{env} only (flatGRPO / judge / progress)|
| `orchestrate_sft_pipeline`   | A         | install → gen → train → eval (4-stage SFT pipeline)            |

The `env_name` parameter ("alfworld" | "webshop") selects which
`Function.from_name("cs224r-hgpo-train-loop", f"train_loop_{env_name}")`
handle to look up at runtime. The two functions accept the same kwarg
signature (both delegate to `_train_loop_impl` in
`infra/app_train_loop.py`), so the orchestrator is genuinely env-agnostic.

## Why .from_name instead of direct import?

The callees live in separately-deployed Modal apps. The clean cross-app
pattern in Modal 1.x is `modal.Function.from_name(app_name,
function_name)`, which requires the target apps to have been **deployed**
(via `modal deploy`) at least once. The cloud launchers
(`scripts/run_*_cloud.sh`) run `modal deploy` for each required app first,
then submit the orchestrator entrypoint.

## Per-round / per-stage sentinels = auto-resume

After every round / stage completes (and `volume.commit()`-ed), the
orchestrator writes a JSON sentinel to the volume. On a preemption-restart
Modal re-invokes the function with the same input args; without the
sentinel scan, every restart would redo R0..R{N-1} from scratch (an
issue that burned ~12h on the first AlfWorld xlbudget run). With the
sentinel scan, the restart resumes at the first incomplete round/stage.
Set `auto_resume=False` to force re-execution.

## Limitations

- The 24-hour timeout caps the maximum total run length. A 10-round
  AlfWorld run takes ~7-12 hours; a 10-round WebShop run takes ~3-4 hr;
  the SFT pipeline takes ~4-6 hr. All comfortably under the 24h cap. If
  you ever need >24 hours, bump `timeout=24 * 60 * 60` accordingly
  (Modal max is 24h on standard plans).
- Each `@app.function` is its own ephemeral container instance — they
  share the orchestrator app id but cannot share in-memory state. All
  cross-stage / cross-round state goes through the Volume sentinels.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

app = modal.App("cs224r-hgpo-orchestrator")


# ============================================================================
# Pattern B: RL with TurnRD (per-round = train_loop + train_turnrd)
# ============================================================================

@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    volumes={VOLUME_MOUNT: volume},
    # 24-hour cap. A standard 10-round AlfWorld run is ~7-12 hr; WebShop
    # ~3-4 hr. Both comfortably under the cap.
    timeout=24 * 60 * 60,
)
def orchestrate_rl_with_turnrd(
    config: str,
    sft_adapter: str,
    env_name: str = "alfworld",
    rounds: int = 10,
    start_round: int = 0,
    episodes_per_round: int = 80,
    eval_episodes: int = 100,
    eval_task_id_base: int = 6500,
    seed: int = 31,
    turnrd_epochs: int = 5,
    turnrd_mode: int = 1,
    turnrd_batch_size: int = 16,
    rollout_temperature: float = 1.0,
    run_name_prefix: str = "TurnRDV2_alfworld_xlbudget",
    adapter_dir: str = "/vol/checkpoints",
    gpu_mem_util: float = 0.4,
    sync_every: int = 8,
    skip_warmup_fit: bool = False,
    auto_resume: bool = True,
) -> dict:
    """Run a multi-round H-GRPO + TurnRDv2 loop cloud-side.

    Per-round shape: (a) `train_loop_{env_name}` collects rollouts, runs
    GRPO updates, writes the replay JSONL via the producer hook, then
    saves the carry-policy adapter; (b) `train_turnrd_run` standalone-fits
    the V-head on the replay buffer and saves the ckpt for the next
    round's reference forward.

    Carry-policy is hard-coded ON (the only mode SOTA recipes use); fork
    if you ever need the reset-each-round mode.

    `replay_path` and `ckpt_path` are NOT orchestrator args — they are
    read from `cfg.turnrd.replay_buffer_path` and `cfg.turnrd.ckpt_path`
    inside the JSON config. The reason: the producer hook inside
    `train_loop_{env_name}` writes the replay JSONL using
    `cfg.turnrd.replay_buffer_path` (NOT a kwarg). If the orchestrator
    accepted these as independent args, a mismatch between args and
    config would mean train_loop writes to one path while train_turnrd
    looks at another — exactly the FileNotFoundError that bit the
    first cloud-orchestrator run. Sourcing from the JSON config is the
    single-source-of-truth fix.

    Args:
      config: path to the JSON config inside the container (typically
        `/workspace/configs/<your_config>.json`).
      sft_adapter: path to the SFT warm-start adapter on the volume.
        Loaded at R0; rounds N>=1 carry the prior round's adapter.
      env_name: "alfworld" | "webshop". Selects which
        `train_loop_<env_name>` function to look up.
      rounds: total round count (default 10).
      start_round: round index to resume from. 0 = fresh run.
      episodes_per_round: H-GRPO episodes per round (default 80).
      eval_episodes: held-out eval pass per round (default 100).
      eval_task_id_base: eval task-id starting offset (default 6500
        — disjoint from training task ranges per the per-seed offset
        math).
      seed: protocol seed; drives the per-seed training task-id slice
        `seed * rounds * episodes_per_round`.
      turnrd_epochs: standalone TurnRD epochs per round (default 5).
      turnrd_mode: 1 = R-prediction (the AlfWorld SOTA path).
      turnrd_batch_size: standalone TurnRD batch size (default 16).
      rollout_temperature: vLLM sampling T for K-trajectory rollouts.
      run_name_prefix: per-round run name prefix; gets `_round{N:02d}`
        and `_seed{S}` appended.
      adapter_dir: directory under which carry-policy adapters are
        written and read (default /vol/checkpoints).
      gpu_mem_util: vLLM GPU memory fraction (default 0.4).
      sync_every: vLLM weight-sync cadence (default 8).
      skip_warmup_fit: when True, skip the standalone TurnRD fit at
        the end of round 0. Used for resume-from-mid-run scenarios.
      auto_resume: when True (default), at startup AND between rounds
        scan the volume for per-round "done" sentinels. Any round whose
        sentinel exists is SKIPPED. This makes the orchestrator
        idempotent under Modal preemption-restarts.

    Returns:
      A summary dict with per-round eval.pct_success and total elapsed.
    """
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/workspace")

    # Modal volume eventual consistency: ensure we see the latest
    # adapters + ckpts written by prior orchestrator turns.
    volume.reload()

    # --- Validate + resolve env routing ---
    if env_name not in ("alfworld", "webshop"):
        raise ValueError(
            f"env_name must be 'alfworld' or 'webshop'; got {env_name!r}"
        )
    train_loop_fn_name = f"train_loop_{env_name}"

    # --- Look up the deployed train_loop + train_turnrd functions ---
    # `from_name` requires `modal deploy` to have been run on each of
    # those app files; the launcher does that before submitting this
    # orchestrator.
    train_loop_fn = modal.Function.from_name(
        "cs224r-hgpo-train-loop", train_loop_fn_name
    )
    train_turnrd_run = modal.Function.from_name(
        "cs224r-hgpo-train-turnrd", "train_turnrd_run"
    )

    # --- Load the JSON config so we can compute per-round task-ids
    #     and (for train_turnrd) extract the arch + aux-loss knobs.
    with open(config) as fh:
        cfg_json = json.load(fh)

    # Single source of truth for replay + ckpt paths.
    turnrd_block = cfg_json.get("turnrd", {}) or {}
    replay_path = str(turnrd_block.get("replay_buffer_path", ""))
    ckpt_path = str(turnrd_block.get("ckpt_path", ""))
    if not replay_path or not ckpt_path:
        raise ValueError(
            f"cfg.turnrd.replay_buffer_path and cfg.turnrd.ckpt_path MUST be "
            f"set in {config}. Got replay_buffer_path={replay_path!r}, "
            f"ckpt_path={ckpt_path!r}."
        )

    k_per_task = int(
        (cfg_json.get("train", {}) or {}).get("K_trajectories_per_task", 4)
    )

    # WebShop env needs a non-zero `num_products` for BM25 index sizing.
    # `_train_loop_impl` falls back to the kwarg ONLY when
    # cfg.env.env_kwargs is empty (which IS the case for all the shipped
    # WebShop SOTA configs). We therefore must NOT pass num_products=0
    # for webshop — that would size the env to 0 products and break the
    # search engine. Resolve from cfg.env.env_kwargs.num_products if
    # present; otherwise pick the env-appropriate default that matches
    # `train_loop_{env_name}`'s own default (webshop=1000, alfworld=0).
    _cfg_env_kwargs_with_turnrd = (
        (cfg_json.get("env", {}) or {}).get("env_kwargs", {}) or {}
    )
    num_products_to_pass = int(_cfg_env_kwargs_with_turnrd.get(
        "num_products",
        1000 if env_name == "webshop" else 0,
    ))

    # Per-seed deterministic task-id slice. Mirrors
    # `OrchestrationConfig.base_task_id_offset` in the local driver.
    base_task_id_offset = int(seed) * int(rounds) * int(episodes_per_round)

    effective_run_name_prefix = f"{run_name_prefix}_seed{int(seed)}"

    turnrd_lr = float(turnrd_block.get("turnrd_lr", 1e-4))

    def _trd_kwarg(key: str, default):
        v = turnrd_block.get(key)
        return v if v is not None else default

    print(f">>> Cloud orchestrator (with-TurnRD) starting")
    print(f"    env_name           : {env_name}")
    print(f"    train_loop fn      : {train_loop_fn_name}")
    print(f"    config             : {config}")
    print(f"    rounds             : {rounds} (start at R{start_round})")
    print(f"    episodes/round     : {episodes_per_round}")
    print(f"    eval/round         : {eval_episodes}")
    print(f"    turnrd_epochs      : {turnrd_epochs}")
    print(f"    seed               : {seed}")
    print(f"    base_task_offset   : {base_task_id_offset}")
    print(f"    train range        : [{base_task_id_offset}, "
          f"{base_task_id_offset + rounds * episodes_per_round})")
    print(f"    eval range         : [{eval_task_id_base}, "
          f"{eval_task_id_base + eval_episodes})")
    print(f"    run-name-prefix    : {effective_run_name_prefix}")
    print(f"    replay (vol)       : {replay_path}")
    print(f"    ckpt   (vol)       : {ckpt_path}")
    print(f"    k_per_task         : {k_per_task}")
    print(f"    gpu_mem_util       : {gpu_mem_util}")
    print(f"    sync_every         : {sync_every}")

    summary: dict = {
        "orchestrator_kind": "rl_with_turnrd",
        "env_name": env_name,
        "rounds_completed": 0,
        "per_round_eval": [],
        "orchestrator_start_ts": time.time(),
    }

    for round_idx in range(start_round, rounds):
        round_t0 = time.time()
        print(f"\n=== Round {round_idx}/{rounds - 1} ===")

        # --- Auto-resume gate ---
        round_done_sentinel = (
            f"{adapter_dir.rstrip('/')}/"
            f"{effective_run_name_prefix}_round{round_idx:02d}_done.json"
        )
        if auto_resume and os.path.exists(round_done_sentinel):
            try:
                with open(round_done_sentinel) as _fh:
                    _prev = json.load(_fh)
                _prev_pct = _prev.get("eval_pct_success", "?")
                _prev_ts = _prev.get("timestamp_iso", "?")
            except Exception:
                _prev_pct = "?"
                _prev_ts = "?"
            print(
                f"    [auto-resume] R{round_idx} SKIPPED — sentinel exists "
                f"({round_done_sentinel}); prior eval_pct_success={_prev_pct}, "
                f"completed_at={_prev_ts}"
            )
            summary["per_round_eval"].append({
                "round_idx": round_idx,
                "elapsed_s": 0.0,
                "eval_pct_success": _prev_pct if isinstance(_prev_pct, (int, float)) else 0.0,
                "eval_truncated": _prev.get("eval_truncated", 0) if isinstance(_prev_pct, (int, float)) else 0,
                "eval_empty_outputs": _prev.get("eval_empty_outputs", 0) if isinstance(_prev_pct, (int, float)) else 0,
                "eval_n_turns_avg": _prev.get("eval_n_turns_avg", 0.0) if isinstance(_prev_pct, (int, float)) else 0.0,
                "auto_resumed": True,
            })
            summary["rounds_completed"] = round_idx - start_round + 1
            continue

        # --- (a) train_loop_{env_name} — rollouts + producer hook
        if round_idx == 0:
            load_adapter = sft_adapter
        else:
            load_adapter = (
                f"{adapter_dir.rstrip('/')}/"
                f"{effective_run_name_prefix}_round{round_idx - 1:02d}_adapter"
            )
        save_adapter_out = (
            f"{adapter_dir.rstrip('/')}/"
            f"{effective_run_name_prefix}_round{round_idx:02d}_adapter"
        )

        task_offset_round = base_task_id_offset + round_idx * episodes_per_round
        run_name_round = f"{effective_run_name_prefix}_round{round_idx:02d}"

        print(f"    [train_loop] task_offset={task_offset_round} "
              f"run_name={run_name_round}")
        print(f"    [train_loop] load_adapter={load_adapter}")
        print(f"    [train_loop] save_adapter_out={save_adapter_out}")

        train_loop_t0 = time.time()
        tl_result = train_loop_fn.remote(
            n_episodes=episodes_per_round,
            k=k_per_task,
            task_id_offset=task_offset_round,
            num_products=num_products_to_pass,
            sync_every=sync_every,
            run_name=run_name_round,
            sft_adapter=load_adapter,
            use_sft_as_ref=True,
            kl_warmup_episodes=0,
            gpu_mem_util=gpu_mem_util,
            config=config,
            eval_episodes=eval_episodes,
            eval_task_id_base=eval_task_id_base,
            round_idx=round_idx,
            save_adapter_out=save_adapter_out,
            rollout_temperature=rollout_temperature,
        )
        train_loop_elapsed = round(time.time() - train_loop_t0, 1)
        eval_pct = float(tl_result.get("eval_pct_success", 0.0))
        eval_n_ok = int(tl_result.get("eval_n_episodes_ok", 0))

        # Reload to see the train_log.json the train_loop wrote.
        volume.reload()
        eval_truncated = 0
        eval_empty = 0
        n_turns_avg = 0.0
        log_path = str(tl_result.get("log_path", "") or "")
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path) as _fh:
                    _full_log = json.load(_fh)
                _ev = _full_log.get("eval", {}) or {}
                eval_truncated = int(_ev.get("truncated", 0))
                eval_empty = int(_ev.get("empty_outputs", 0))
                n_turns_avg = float(_ev.get("n_turns_avg", 0.0))
            except Exception as _ex:
                print(
                    f"    [warn] could not read full eval block from "
                    f"{log_path!r}: {_ex!r}"
                )
        print(
            f"    [train_loop] R{round_idx} DONE in {train_loop_elapsed}s | "
            f"eval.pct_success={eval_pct:.3f} | "
            f"eval.n_episodes_ok={eval_n_ok} | "
            f"eval.truncated={eval_truncated} | "
            f"eval.n_turns_avg={n_turns_avg:.1f} | "
            f"eval.empty_outputs={eval_empty}"
        )

        volume.commit()

        # --- (b) train_turnrd — standalone V-head fit on the replay
        if round_idx == 0 and skip_warmup_fit:
            print(f"    [train_turnrd] R0 SKIPPED per --skip-warmup-fit")
        else:
            train_turnrd_t0 = time.time()
            tt_kwargs = dict(
                replay=replay_path,
                mode=turnrd_mode,
                n_epochs=turnrd_epochs,
                batch_size=turnrd_batch_size,
                lr=turnrd_lr,
                ckpt_out=ckpt_path,
                version=str(_trd_kwarg("version", "v1")),
                layers=int(_trd_kwarg("layers", 6)),
                hidden_size=int(_trd_kwarg("hidden_size", 384)),
                n_heads=int(_trd_kwarg("n_heads", 4)),
                max_turns=int(_trd_kwarg("max_turns", 64)),
                dropout=float(_trd_kwarg("dropout", 0.1)),
                progress_prior_strength=float(
                    _trd_kwarg("progress_prior_strength", 1.0)
                ),
                lambda_value=float(_trd_kwarg("lambda_value", 0.5)),
                lambda_rank=float(_trd_kwarg("lambda_rank", 0.1)),
                lambda_progress=float(_trd_kwarg("lambda_progress", 0.01)),
                rank_margin=float(_trd_kwarg("rank_margin", 0.1)),
                recency_decay_half_life=float(
                    _trd_kwarg("recency_decay_half_life", 0.0)
                ),
                legacy_decay_weight=float(
                    _trd_kwarg("legacy_decay_weight", 0.5)
                ),
                min_batch_weight=float(
                    _trd_kwarg("min_batch_weight", 1e-3)
                ),
                goal_match_blend=float(
                    _trd_kwarg("goal_match_blend", 0.0)
                ),
            )
            tt_result = train_turnrd_run.remote(**tt_kwargs)
            train_turnrd_elapsed = round(time.time() - train_turnrd_t0, 1)
            final_loss = float(tt_result.get("final_loss", float("nan")))
            n_steps = int(tt_result.get("n_steps", 0))
            print(
                f"    [train_turnrd] R{round_idx} DONE in "
                f"{train_turnrd_elapsed}s | final_loss={final_loss:.4f} | "
                f"n_steps={n_steps}"
            )
            volume.commit()

        round_elapsed = round(time.time() - round_t0, 1)
        summary["per_round_eval"].append({
            "round_idx": round_idx,
            "elapsed_s": round_elapsed,
            "eval_pct_success": eval_pct,
            "eval_truncated": eval_truncated,
            "eval_empty_outputs": eval_empty,
            "eval_n_turns_avg": n_turns_avg,
        })
        summary["rounds_completed"] = round_idx - start_round + 1
        print(f"    Round {round_idx} total elapsed: {round_elapsed}s")

        # --- Write the per-round "done" sentinel ---
        import datetime as _dt
        sentinel_payload = {
            "round_idx": round_idx,
            "timestamp": time.time(),
            "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "eval_pct_success": eval_pct,
            "eval_n_episodes_ok": eval_n_ok,
            "eval_truncated": eval_truncated,
            "eval_empty_outputs": eval_empty,
            "eval_n_turns_avg": n_turns_avg,
            "round_elapsed_s": round_elapsed,
            "adapter_saved_at": save_adapter_out,
            "skip_warmup_fit_used": bool(round_idx == 0 and skip_warmup_fit),
            "env_name": env_name,
        }
        os.makedirs(adapter_dir.rstrip("/"), exist_ok=True)
        with open(round_done_sentinel, "w") as _fh:
            json.dump(sentinel_payload, _fh, indent=2)
        volume.commit()
        print(f"    [auto-resume] R{round_idx} sentinel written: {round_done_sentinel}")

    summary["orchestrator_end_ts"] = time.time()
    summary["total_elapsed_s"] = round(
        summary["orchestrator_end_ts"] - summary["orchestrator_start_ts"], 1
    )
    print(
        f"\n=== Done. {summary['rounds_completed']} rounds × "
        f"{episodes_per_round} episodes = "
        f"{summary['rounds_completed'] * episodes_per_round} total H-GRPO "
        f"episodes in {summary['total_elapsed_s']}s ==="
    )

    summary_dir = f"/vol/manifests/{effective_run_name_prefix}_orchestrator_summary"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = f"{summary_dir}/summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f">>> Orchestrator summary written: {summary_path}")
    volume.commit()

    return summary


# Backward-compat alias for in-flight ephemeral runs that hold a
# `Function.from_name` handle by the old name. Module-level aliasing is
# safe because the underlying registered function is the same one;
# Python imports also resolve `orchestrate_alfworld_rl` to the new impl.
# Note: `modal run infra/app_orchestrator.py::orchestrate_alfworld_rl` is
# NOT supported via this alias — Modal looks the function up by its
# decorator-registered name (`orchestrate_rl_with_turnrd`). Update the
# in-flight launcher to use the new name + `--env-name alfworld`.
orchestrate_alfworld_rl = orchestrate_rl_with_turnrd


# ============================================================================
# Pattern C: RL without TurnRD (per-round = train_loop only)
# ============================================================================

@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    volumes={VOLUME_MOUNT: volume},
    timeout=24 * 60 * 60,
)
def orchestrate_rl_no_turnrd(
    config: str,
    sft_adapter: str,
    env_name: str = "webshop",
    rounds: int = 10,
    start_round: int = 0,
    episodes_per_round: int = 80,
    eval_episodes: int = 100,
    eval_task_id_base: int = 6500,
    seed: int = 23,
    rollout_temperature: float = 1.0,
    run_name_prefix: str = "webshop_flatGRPO_v1",
    adapter_dir: str = "/vol/checkpoints",
    gpu_mem_util: float = 0.4,
    sync_every: int = 8,
    auto_resume: bool = True,
) -> dict:
    """Run a multi-round H-GRPO loop cloud-side WITHOUT the TurnRD step.

    Per-round shape: train_loop_{env_name} only (no standalone V-head fit).
    Used by Method C / FlatGRPO / LLMJudge / Progress recipes — any
    decomposer that doesn't need a TurnRD model trained between rounds.

    Carry-policy adapter chaining matches the with-TurnRD variant:
    R0 loads sft_adapter; R_N>0 loads R_{N-1}'s saved adapter. Per-round
    sentinels enable auto-resume on preemption-restart.

    `gpu_mem_util` and `sync_every` are honored if the JSON config does
    NOT specify train.gpu_mem_util / train.sync_every; otherwise the
    JSON values win (matching `_train_loop_impl`'s precedence).

    Args:
      config: path to the JSON config inside the container.
      sft_adapter: warm-start adapter on the volume (loaded at R0).
      env_name: "alfworld" | "webshop".
      rounds: total round count.
      start_round: round to resume from (0 = fresh).
      episodes_per_round: H-GRPO episodes per round.
      eval_episodes: held-out eval pass per round.
      eval_task_id_base: eval task-id starting offset.
      seed: protocol seed; drives the per-seed training task-id slice.
      rollout_temperature: vLLM sampling T.
      run_name_prefix: per-round run name prefix; gets `_round{N:02d}`
        and `_seed{S}` appended.
      adapter_dir: directory for carry-policy adapters.
      gpu_mem_util: vLLM GPU memory fraction.
      sync_every: vLLM weight-sync cadence.
      auto_resume: scan + skip rounds whose "done" sentinel exists.

    Returns:
      A summary dict with per-round eval.pct_success and total elapsed.
    """
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/workspace")
    volume.reload()

    if env_name not in ("alfworld", "webshop"):
        raise ValueError(
            f"env_name must be 'alfworld' or 'webshop'; got {env_name!r}"
        )
    train_loop_fn_name = f"train_loop_{env_name}"

    train_loop_fn = modal.Function.from_name(
        "cs224r-hgpo-train-loop", train_loop_fn_name
    )

    with open(config) as fh:
        cfg_json = json.load(fh)

    k_per_task = int(
        (cfg_json.get("train", {}) or {}).get("K_trajectories_per_task", 4)
    )

    # WebShop env needs a non-zero `num_products` for BM25 index sizing
    # (see orchestrate_rl_with_turnrd for the full rationale).
    _cfg_env_kwargs_no_turnrd = (
        (cfg_json.get("env", {}) or {}).get("env_kwargs", {}) or {}
    )
    num_products_to_pass = int(_cfg_env_kwargs_no_turnrd.get(
        "num_products",
        1000 if env_name == "webshop" else 0,
    ))

    base_task_id_offset = int(seed) * int(rounds) * int(episodes_per_round)
    effective_run_name_prefix = f"{run_name_prefix}_seed{int(seed)}"

    print(f">>> Cloud orchestrator (no-TurnRD) starting")
    print(f"    env_name           : {env_name}")
    print(f"    train_loop fn      : {train_loop_fn_name}")
    print(f"    config             : {config}")
    print(f"    rounds             : {rounds} (start at R{start_round})")
    print(f"    episodes/round     : {episodes_per_round}")
    print(f"    eval/round         : {eval_episodes}")
    print(f"    seed               : {seed}")
    print(f"    base_task_offset   : {base_task_id_offset}")
    print(f"    train range        : [{base_task_id_offset}, "
          f"{base_task_id_offset + rounds * episodes_per_round})")
    print(f"    eval range         : [{eval_task_id_base}, "
          f"{eval_task_id_base + eval_episodes})")
    print(f"    run-name-prefix    : {effective_run_name_prefix}")
    print(f"    k_per_task         : {k_per_task}")
    print(f"    gpu_mem_util       : {gpu_mem_util}")
    print(f"    sync_every         : {sync_every}")

    summary: dict = {
        "orchestrator_kind": "rl_no_turnrd",
        "env_name": env_name,
        "rounds_completed": 0,
        "per_round_eval": [],
        "orchestrator_start_ts": time.time(),
    }

    for round_idx in range(start_round, rounds):
        round_t0 = time.time()
        print(f"\n=== Round {round_idx}/{rounds - 1} ===")

        round_done_sentinel = (
            f"{adapter_dir.rstrip('/')}/"
            f"{effective_run_name_prefix}_round{round_idx:02d}_done.json"
        )
        if auto_resume and os.path.exists(round_done_sentinel):
            try:
                with open(round_done_sentinel) as _fh:
                    _prev = json.load(_fh)
                _prev_pct = _prev.get("eval_pct_success", "?")
                _prev_ts = _prev.get("timestamp_iso", "?")
            except Exception:
                _prev_pct = "?"
                _prev_ts = "?"
            print(
                f"    [auto-resume] R{round_idx} SKIPPED — sentinel exists "
                f"({round_done_sentinel}); prior eval_pct_success={_prev_pct}, "
                f"completed_at={_prev_ts}"
            )
            summary["per_round_eval"].append({
                "round_idx": round_idx,
                "elapsed_s": 0.0,
                "eval_pct_success": _prev_pct if isinstance(_prev_pct, (int, float)) else 0.0,
                "eval_truncated": _prev.get("eval_truncated", 0) if isinstance(_prev_pct, (int, float)) else 0,
                "eval_empty_outputs": _prev.get("eval_empty_outputs", 0) if isinstance(_prev_pct, (int, float)) else 0,
                "eval_n_turns_avg": _prev.get("eval_n_turns_avg", 0.0) if isinstance(_prev_pct, (int, float)) else 0.0,
                "auto_resumed": True,
            })
            summary["rounds_completed"] = round_idx - start_round + 1
            continue

        if round_idx == 0:
            load_adapter = sft_adapter
        else:
            load_adapter = (
                f"{adapter_dir.rstrip('/')}/"
                f"{effective_run_name_prefix}_round{round_idx - 1:02d}_adapter"
            )
        save_adapter_out = (
            f"{adapter_dir.rstrip('/')}/"
            f"{effective_run_name_prefix}_round{round_idx:02d}_adapter"
        )

        task_offset_round = base_task_id_offset + round_idx * episodes_per_round
        run_name_round = f"{effective_run_name_prefix}_round{round_idx:02d}"

        print(f"    [train_loop] task_offset={task_offset_round} "
              f"run_name={run_name_round}")
        print(f"    [train_loop] load_adapter={load_adapter}")
        print(f"    [train_loop] save_adapter_out={save_adapter_out}")

        train_loop_t0 = time.time()
        tl_result = train_loop_fn.remote(
            n_episodes=episodes_per_round,
            k=k_per_task,
            task_id_offset=task_offset_round,
            num_products=num_products_to_pass,
            sync_every=sync_every,
            run_name=run_name_round,
            sft_adapter=load_adapter,
            use_sft_as_ref=True,
            kl_warmup_episodes=0,
            gpu_mem_util=gpu_mem_util,
            config=config,
            eval_episodes=eval_episodes,
            eval_task_id_base=eval_task_id_base,
            round_idx=round_idx,
            save_adapter_out=save_adapter_out,
            rollout_temperature=rollout_temperature,
        )
        train_loop_elapsed = round(time.time() - train_loop_t0, 1)
        eval_pct = float(tl_result.get("eval_pct_success", 0.0))
        eval_n_ok = int(tl_result.get("eval_n_episodes_ok", 0))

        volume.reload()
        eval_truncated = 0
        eval_empty = 0
        n_turns_avg = 0.0
        log_path = str(tl_result.get("log_path", "") or "")
        if log_path and os.path.exists(log_path):
            try:
                with open(log_path) as _fh:
                    _full_log = json.load(_fh)
                _ev = _full_log.get("eval", {}) or {}
                eval_truncated = int(_ev.get("truncated", 0))
                eval_empty = int(_ev.get("empty_outputs", 0))
                n_turns_avg = float(_ev.get("n_turns_avg", 0.0))
            except Exception as _ex:
                print(
                    f"    [warn] could not read full eval block from "
                    f"{log_path!r}: {_ex!r}"
                )
        print(
            f"    [train_loop] R{round_idx} DONE in {train_loop_elapsed}s | "
            f"eval.pct_success={eval_pct:.3f} | "
            f"eval.n_episodes_ok={eval_n_ok} | "
            f"eval.truncated={eval_truncated} | "
            f"eval.n_turns_avg={n_turns_avg:.1f} | "
            f"eval.empty_outputs={eval_empty}"
        )

        volume.commit()

        round_elapsed = round(time.time() - round_t0, 1)
        summary["per_round_eval"].append({
            "round_idx": round_idx,
            "elapsed_s": round_elapsed,
            "eval_pct_success": eval_pct,
            "eval_truncated": eval_truncated,
            "eval_empty_outputs": eval_empty,
            "eval_n_turns_avg": n_turns_avg,
        })
        summary["rounds_completed"] = round_idx - start_round + 1
        print(f"    Round {round_idx} total elapsed: {round_elapsed}s")

        # Sentinel — same shape as the with-TurnRD path, sans
        # final_loss / n_steps fields. Status-table scripts that read
        # these JSONs handle the absent keys gracefully.
        import datetime as _dt
        sentinel_payload = {
            "round_idx": round_idx,
            "timestamp": time.time(),
            "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "eval_pct_success": eval_pct,
            "eval_n_episodes_ok": eval_n_ok,
            "eval_truncated": eval_truncated,
            "eval_empty_outputs": eval_empty,
            "eval_n_turns_avg": n_turns_avg,
            "round_elapsed_s": round_elapsed,
            "adapter_saved_at": save_adapter_out,
            "env_name": env_name,
            "orchestrator_kind": "rl_no_turnrd",
        }
        os.makedirs(adapter_dir.rstrip("/"), exist_ok=True)
        with open(round_done_sentinel, "w") as _fh:
            json.dump(sentinel_payload, _fh, indent=2)
        volume.commit()
        print(f"    [auto-resume] R{round_idx} sentinel written: {round_done_sentinel}")

    summary["orchestrator_end_ts"] = time.time()
    summary["total_elapsed_s"] = round(
        summary["orchestrator_end_ts"] - summary["orchestrator_start_ts"], 1
    )
    print(
        f"\n=== Done. {summary['rounds_completed']} rounds × "
        f"{episodes_per_round} episodes = "
        f"{summary['rounds_completed'] * episodes_per_round} total H-GRPO "
        f"episodes in {summary['total_elapsed_s']}s ==="
    )

    summary_dir = f"/vol/manifests/{effective_run_name_prefix}_orchestrator_summary"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = f"{summary_dir}/summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f">>> Orchestrator summary written: {summary_path}")
    volume.commit()

    return summary


# ============================================================================
# Pattern A: WebShop SFT pipeline (4-stage: install → gen → train → eval)
# ============================================================================

# Stage ordering for the SFT pipeline. The orchestrator iterates through
# this list in order and skips any stage whose sentinel exists. The MODE
# flag gates which stages are eligible to run.
_SFT_STAGES: tuple[str, ...] = ("1a", "1b", "1c", "2", "3", "4")

# Stages that each MODE is allowed to execute. Mirrors the bash launcher
# `scripts/run_webshop_sft_v3_mlpr32.sh`'s MODE gating.
_SFT_MODE_STAGES: dict[str, set[str]] = {
    "full": {"1a", "1b", "1c", "2", "3", "4"},
    "skip-install": {"2", "3", "4"},
    "skip-gen": {"3", "4"},
    "train-only": {"3", "4"},
    "eval-only": {"4"},
}


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    volumes={VOLUME_MOUNT: volume},
    # SFT pipeline takes ~4-6 hr end-to-end (install ~20 min + gen ~30 min
    # + train ~3-5 hr + eval ~30 min). 24h cap is comfortable.
    timeout=24 * 60 * 60,
)
def orchestrate_sft_pipeline(
    run_name: str,
    mode: str = "full",
    n_sessions: int = 2000,
    include_human_trajs: bool = True,
    human_trajs_min_reward: float = 0.5,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.99,
    epochs: int = 6,
    learning_rate: float = 5e-5,
    max_seq_len: int = 2048,
    grad_accum: int = 8,
    min_reward: float = 0.5,
    lora_rank: int = 32,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    eval_episodes: int = 200,
    eval_task_id_base: int = 6500,
    data_path: str = "/vol/data/webshop/oracle_trajs.jsonl",
    adapter_path: str = "",
    sentinel_dir: str = "/vol/checkpoints",
    eval_config: str = "/workspace/configs/SFTOnly_webshop_mlpr32.json",
    eval_gpu_mem_util: float = 0.30,
    auto_resume: bool = True,
) -> dict:
    """Run the 4-stage WebShop SFT pipeline cloud-side.

    Stages (each gated on a sentinel file):
      1a: pip_install_webshop (cs224r-hgpo-webshop-install)
      1b: download_spacy_model (cs224r-hgpo-webshop-install)
      1c: build_index_1k (cs224r-hgpo-webshop-install)
      2:  generate_sft_trajectories (cs224r-hgpo-webshop-sft-gen)
      3:  sft_train (cs224r-hgpo-sft-train)
      4:  train_loop_webshop in eval-only mode (cs224r-hgpo-train-loop)

    There is no dedicated `eval_sft` Modal function; Stage 4 reuses
    `train_loop_webshop` with `n_episodes=0` exactly as the bash launcher
    `scripts/run_webshop_sft_v3_mlpr32.sh` does.

    The `sft_train` function writes its adapter to a timestamped dir
    `/vol/checkpoints/<run_name>_<ts>` — we resolve the actual dir AFTER
    Stage 3 by scanning `/vol/checkpoints/` for `<run_name>_*` and
    picking the newest. Override via `adapter_path` if you want to
    force-eval a specific adapter (handy for `mode=eval-only`).

    Args:
      run_name: SFT run name (`sft_train`'s `run_name` arg; also used as
        prefix for stage sentinels). Should match the bash launcher's
        RUN_NAME knob.
      mode: "full" | "skip-install" | "skip-gen" | "train-only" | "eval-only".
        Selects which stages are eligible to run. Already-completed stages
        (per sentinel scan) are always skipped when `auto_resume=True`.
      n_sessions, include_human_trajs, human_trajs_min_reward,
        max_result_pages, max_steps_per_episode, reward_threshold:
        forwarded to `generate_sft_trajectories`.
      epochs, learning_rate, max_seq_len, grad_accum, min_reward,
        lora_rank, lora_target_modules: forwarded to `sft_train`.
      eval_episodes, eval_task_id_base: forwarded to Stage 4 train_loop.
      data_path: where gen writes / train reads the SFT JSONL.
      adapter_path: optional override for Stage 4's `sft_adapter` arg.
        When empty (default), Stage 4 resolves it by globbing
        `/vol/checkpoints/<run_name>_*` and picking newest.
      sentinel_dir: where per-stage "done" sentinels are written.
      eval_config: container path to the SFTOnly_webshop_mlpr32.json
        config used by Stage 4.
      eval_gpu_mem_util: vLLM mem util for Stage 4 (default 0.30).
      auto_resume: scan + skip stages whose "done" sentinel exists.

    Returns:
      A summary dict with per-stage status and total elapsed.
    """
    import datetime as _dt
    import json
    import os
    import sys
    import time

    sys.path.insert(0, "/workspace")
    volume.reload()

    if mode not in _SFT_MODE_STAGES:
        raise ValueError(
            f"mode must be one of {sorted(_SFT_MODE_STAGES)}; got {mode!r}"
        )
    allowed_stages = _SFT_MODE_STAGES[mode]

    def _sentinel_path(stage: str) -> str:
        return (
            f"{sentinel_dir.rstrip('/')}/"
            f"sft_pipeline_{run_name}_stage{stage}_done.json"
        )

    def _write_sentinel(stage: str, payload: dict) -> None:
        os.makedirs(sentinel_dir.rstrip("/"), exist_ok=True)
        with open(_sentinel_path(stage), "w") as _fh:
            json.dump({
                "stage": stage,
                "run_name": run_name,
                "timestamp": time.time(),
                "timestamp_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                **payload,
            }, _fh, indent=2, default=str)
        volume.commit()

    def _is_done(stage: str) -> bool:
        return auto_resume and os.path.exists(_sentinel_path(stage))

    # Look up all Modal apps' functions we will dispatch.
    fn_pip = modal.Function.from_name(
        "cs224r-hgpo-webshop-install", "pip_install_webshop"
    )
    fn_spacy = modal.Function.from_name(
        "cs224r-hgpo-webshop-install", "download_spacy_model"
    )
    fn_index = modal.Function.from_name(
        "cs224r-hgpo-webshop-install", "build_index_1k"
    )
    fn_gen = modal.Function.from_name(
        "cs224r-hgpo-webshop-sft-gen", "generate_sft_trajectories"
    )
    fn_sft = modal.Function.from_name(
        "cs224r-hgpo-sft-train", "sft_train"
    )
    fn_eval = modal.Function.from_name(
        "cs224r-hgpo-train-loop", "train_loop_webshop"
    )

    print(f">>> Cloud SFT pipeline starting")
    print(f"    run_name           : {run_name}")
    print(f"    mode               : {mode}")
    print(f"    eligible stages    : {sorted(allowed_stages)}")
    print(f"    auto_resume        : {auto_resume}")
    print(f"    data_path          : {data_path}")
    print(f"    sentinel_dir       : {sentinel_dir}")
    print(f"    eval_config        : {eval_config}")
    print(f"    lora_rank          : {lora_rank}")
    print(f"    lora_target_modules: {lora_target_modules}")

    summary: dict = {
        "orchestrator_kind": "sft_pipeline",
        "run_name": run_name,
        "mode": mode,
        "stages": {},
        "orchestrator_start_ts": time.time(),
    }

    def _maybe_run(stage: str, label: str, fn_call) -> dict:
        """Run a stage if mode allows and sentinel absent; record summary."""
        if stage not in allowed_stages:
            print(f"\n>>> Stage {stage} ({label}) SKIPPED — not in mode={mode}")
            summary["stages"][stage] = {"status": "skipped_by_mode"}
            return {}
        if _is_done(stage):
            print(
                f"\n>>> Stage {stage} ({label}) SKIPPED — sentinel exists "
                f"({_sentinel_path(stage)})"
            )
            try:
                with open(_sentinel_path(stage)) as _fh:
                    prev = json.load(_fh)
            except Exception:
                prev = {}
            summary["stages"][stage] = {"status": "skipped_auto_resume", **prev}
            return prev
        print(f"\n>>> Stage {stage} ({label}) STARTING")
        t0 = time.time()
        result = fn_call()
        elapsed = round(time.time() - t0, 1)
        print(f">>> Stage {stage} ({label}) DONE in {elapsed}s")
        # Reload so the next stage sees the just-written files.
        volume.reload()
        payload = {
            "elapsed_s": elapsed,
            "result_keys": sorted(list(result.keys())) if isinstance(result, dict) else None,
            "result_preview": {k: v for k, v in (result.items() if isinstance(result, dict) else [])
                               if isinstance(v, (int, float, str, bool))},
        }
        _write_sentinel(stage, payload)
        summary["stages"][stage] = {"status": "completed", "elapsed_s": elapsed}
        return result

    # Stage 1a: pip install
    _maybe_run("1a", "pip_install_webshop", lambda: fn_pip.remote())

    # Stage 1b: spaCy model
    _maybe_run("1b", "download_spacy_model", lambda: fn_spacy.remote())

    # Stage 1c: BM25 index
    _maybe_run("1c", "build_index_1k", lambda: fn_index.remote())

    # Stage 2: generate oracle SFT trajectories
    _maybe_run("2", "generate_sft_trajectories", lambda: fn_gen.remote(
        n_sessions=n_sessions,
        output_path=data_path,
        max_result_pages=max_result_pages,
        max_steps_per_episode=max_steps_per_episode,
        reward_threshold=reward_threshold,
        include_human_trajs=include_human_trajs,
        human_trajs_min_reward=human_trajs_min_reward,
    ))

    # Stage 3: SFT train. `sft_train` writes adapter to
    # `/vol/checkpoints/<run_name>_<ts>`; we capture the returned dict's
    # ckpt_dir so Stage 4 can find it without globbing.
    sft_result = _maybe_run("3", "sft_train", lambda: fn_sft.remote(
        epochs=epochs,
        learning_rate=learning_rate,
        min_reward=min_reward,
        max_seq_len=max_seq_len,
        grad_accum=grad_accum,
        run_name=run_name,
        data_path=data_path,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
    ))

    # Resolve adapter path for Stage 4:
    # 1. Explicit `adapter_path` arg wins (handy for eval-only mode).
    # 2. Else use ckpt_dir from sft_train's return (or its skipped
    #    sentinel's result_preview).
    # 3. Else glob `/vol/checkpoints/<run_name>_*` and pick newest dir.
    resolved_adapter = adapter_path.strip()
    if not resolved_adapter:
        if isinstance(sft_result, dict):
            ckpt_dir = sft_result.get("ckpt_dir") or (
                (sft_result.get("result_preview") or {}).get("ckpt_dir")
            )
            if ckpt_dir:
                resolved_adapter = str(ckpt_dir)
    if not resolved_adapter:
        import glob
        candidates = sorted(
            glob.glob(f"/vol/checkpoints/{run_name}_*"),
            reverse=True,
        )
        if candidates:
            resolved_adapter = candidates[0]
    if not resolved_adapter and "4" in allowed_stages and not _is_done("4"):
        raise RuntimeError(
            f"Could not resolve adapter path for Stage 4. Tried "
            f"adapter_path arg (empty), sft_train ckpt_dir (no run), and "
            f"glob /vol/checkpoints/{run_name}_* (no matches). Pass "
            f"adapter_path=... explicitly or re-run with --mode=full."
        )

    eval_ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_run_name = f"SFTOnly_webshop_mlpr32_eval_{eval_ts}"

    # Stage 4: eval-only train_loop_webshop (n_episodes=0).
    _maybe_run("4", "eval (train_loop_webshop n_episodes=0)", lambda: fn_eval.remote(
        config=eval_config,
        n_episodes=0,
        eval_episodes=eval_episodes,
        eval_task_id_base=eval_task_id_base,
        sft_adapter=resolved_adapter,
        gpu_mem_util=eval_gpu_mem_util,
        run_name=eval_run_name,
    ))

    summary["resolved_adapter"] = resolved_adapter
    summary["eval_run_name"] = eval_run_name
    summary["orchestrator_end_ts"] = time.time()
    summary["total_elapsed_s"] = round(
        summary["orchestrator_end_ts"] - summary["orchestrator_start_ts"], 1
    )
    print(
        f"\n=== SFT pipeline done. total elapsed={summary['total_elapsed_s']}s ==="
    )

    summary_dir = f"/vol/manifests/sft_pipeline_{run_name}_orchestrator_summary"
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = f"{summary_dir}/summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f">>> SFT pipeline summary written: {summary_path}")
    volume.commit()

    return summary


# ============================================================================
# Local entrypoints (one per orchestrator function)
# ============================================================================

@app.local_entrypoint()
def main(
    config: str,
    sft_adapter: str,
    env_name: str = "alfworld",
    rounds: int = 10,
    start_round: int = 0,
    episodes_per_round: int = 80,
    eval_episodes: int = 100,
    eval_task_id_base: int = 6500,
    seed: int = 31,
    turnrd_epochs: int = 5,
    turnrd_mode: int = 1,
    turnrd_batch_size: int = 16,
    rollout_temperature: float = 1.0,
    run_name_prefix: str = "TurnRDV2_alfworld_xlbudget",
    adapter_dir: str = "/vol/checkpoints",
    gpu_mem_util: float = 0.4,
    sync_every: int = 8,
    skip_warmup_fit: bool = False,
    auto_resume: bool = True,
) -> None:
    """Local entrypoint for the with-TurnRD orchestrator. Mirrors the
    legacy `orchestrate_alfworld_rl` CLI exactly, with the addition of
    `--env-name {alfworld,webshop}` (default `alfworld` preserves
    pre-rename behavior). For `--detach` use:

      modal run --detach infra/app_orchestrator.py::orchestrate_rl_with_turnrd \\
        --config /workspace/configs/foo.json --sft-adapter /vol/checkpoints/... \\
        --env-name alfworld --rounds 10 ...
    """
    import json as _json

    result = orchestrate_rl_with_turnrd.remote(
        config=config,
        sft_adapter=sft_adapter,
        env_name=env_name,
        rounds=rounds,
        start_round=start_round,
        episodes_per_round=episodes_per_round,
        eval_episodes=eval_episodes,
        eval_task_id_base=eval_task_id_base,
        seed=seed,
        turnrd_epochs=turnrd_epochs,
        turnrd_mode=turnrd_mode,
        turnrd_batch_size=turnrd_batch_size,
        rollout_temperature=rollout_temperature,
        run_name_prefix=run_name_prefix,
        adapter_dir=adapter_dir,
        gpu_mem_util=gpu_mem_util,
        sync_every=sync_every,
        skip_warmup_fit=skip_warmup_fit,
        auto_resume=auto_resume,
    )
    print(_json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def main_no_turnrd(
    config: str,
    sft_adapter: str,
    env_name: str = "webshop",
    rounds: int = 10,
    start_round: int = 0,
    episodes_per_round: int = 80,
    eval_episodes: int = 100,
    eval_task_id_base: int = 6500,
    seed: int = 23,
    rollout_temperature: float = 1.0,
    run_name_prefix: str = "webshop_flatGRPO_v1",
    adapter_dir: str = "/vol/checkpoints",
    gpu_mem_util: float = 0.4,
    sync_every: int = 8,
    auto_resume: bool = True,
) -> None:
    """Local entrypoint for the no-TurnRD orchestrator. Submit via:

      modal run --detach infra/app_orchestrator.py::orchestrate_rl_no_turnrd \\
        --config /workspace/configs/foo.json --sft-adapter /vol/... \\
        --env-name webshop --rounds 10 ...

    (Or use this local_entrypoint via `infra/app_orchestrator.py::main_no_turnrd`.)
    """
    import json as _json

    result = orchestrate_rl_no_turnrd.remote(
        config=config,
        sft_adapter=sft_adapter,
        env_name=env_name,
        rounds=rounds,
        start_round=start_round,
        episodes_per_round=episodes_per_round,
        eval_episodes=eval_episodes,
        eval_task_id_base=eval_task_id_base,
        seed=seed,
        rollout_temperature=rollout_temperature,
        run_name_prefix=run_name_prefix,
        adapter_dir=adapter_dir,
        gpu_mem_util=gpu_mem_util,
        sync_every=sync_every,
        auto_resume=auto_resume,
    )
    print(_json.dumps(result, indent=2, default=str))


@app.local_entrypoint()
def main_sft_pipeline(
    run_name: str,
    mode: str = "full",
    n_sessions: int = 2000,
    include_human_trajs: bool = True,
    human_trajs_min_reward: float = 0.5,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.99,
    epochs: int = 6,
    learning_rate: float = 5e-5,
    max_seq_len: int = 2048,
    grad_accum: int = 8,
    min_reward: float = 0.5,
    lora_rank: int = 32,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    eval_episodes: int = 200,
    eval_task_id_base: int = 6500,
    data_path: str = "/vol/data/webshop/oracle_trajs.jsonl",
    adapter_path: str = "",
    sentinel_dir: str = "/vol/checkpoints",
    eval_config: str = "/workspace/configs/SFTOnly_webshop_mlpr32.json",
    eval_gpu_mem_util: float = 0.30,
    auto_resume: bool = True,
) -> None:
    """Local entrypoint for the SFT pipeline orchestrator. Submit via:

      modal run --detach infra/app_orchestrator.py::orchestrate_sft_pipeline \\
        --run-name sft_webshop_v3_mlpr32_<ts> --mode full --n-sessions 2000 ...
    """
    import json as _json

    result = orchestrate_sft_pipeline.remote(
        run_name=run_name,
        mode=mode,
        n_sessions=n_sessions,
        include_human_trajs=include_human_trajs,
        human_trajs_min_reward=human_trajs_min_reward,
        max_result_pages=max_result_pages,
        max_steps_per_episode=max_steps_per_episode,
        reward_threshold=reward_threshold,
        epochs=epochs,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        grad_accum=grad_accum,
        min_reward=min_reward,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        eval_episodes=eval_episodes,
        eval_task_id_base=eval_task_id_base,
        data_path=data_path,
        adapter_path=adapter_path,
        sentinel_dir=sentinel_dir,
        eval_config=eval_config,
        eval_gpu_mem_util=eval_gpu_mem_util,
        auto_resume=auto_resume,
    )
    print(_json.dumps(result, indent=2, default=str))
