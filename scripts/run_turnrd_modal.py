#!/usr/bin/env python3
"""Method-B (TurnRD) end-to-end orchestration on Modal.

Coordinates the **producer** (parent H-GRPO `train_loop`) and the
**standalone TurnRD trainer** in a round-robin so the H-GRPO policy
keeps training while TurnRD's reward decomposition is periodically
re-fitted from the accumulated replay buffer.

  Round 0: warm-up
    - train_loop emits replay rows (no TurnRD ckpt yet → trainer's
      refresh_fn logs a warning and skips the load; the in-memory
      decomposer trains from random init).
    - train_turnrd then fits TurnRD on the accumulated rows and writes
      the first ckpt.
  Round i ≥ 1: refresh-driven
    - train_loop runs `--episodes-per-round` more episodes. The
      trainer's refresh_fn (built by build_trainer_from_config when
      `cfg.turnrd.ckpt_path` is set) loads the previous round's ckpt
      at the configured cadence, so every group post-load uses the
      latest decomposer.
    - train_turnrd re-fits on the (now larger) replay buffer.

The `/vol/...` paths are shared across Modal container instances via
the cs224r project Volume — see `infra/common.py::volume`.

Usage:
  scripts/run_turnrd_modal.py --rounds 5 --episodes-per-round 40
  scripts/run_turnrd_modal.py --dry-run    # print commands only
  scripts/run_turnrd_modal.py --rounds 1 --episodes-per-round 2 --turnrd-epochs 1   # tiny smoke

Wall-clock budget (real protocol, real WebShop, K=4 trajectories):
  Per round: ~12-15 min train_loop + ~30 s standalone fit ≈ 13-16 min.
  5-round protocol: ~65-80 min total (assume 90 min for safety).

  → If running from a devmate session, set the execute_command timeout
    to >= 7200000 ms (2 hours). The previous 60-min cap killed the
    LOCAL polling subprocess at the boundary between Round 3 and
    Round 4 — note that this only kills the local poller; the cloud
    jobs themselves are detached so they continue, but the
    orchestrator stops submitting subsequent rounds.
  → For long unattended runs, prefer running this script under
    `nohup` in a separate terminal so its lifetime is bounded only
    by the actual orchestration, not by any IDE/agent session
    timeout. Example:
      nohup scripts/run_turnrd_modal.py --rounds 5 \
        --episodes-per-round 40 --turnrd-epochs 3 --seed 11 \
        --sft-adapter /vol/checkpoints/sft_v3_<ts> \
        > /tmp/protocol_seed11.log 2>&1 &

Requires `modal` on PATH. The two Modal app entrypoints invoked are:
  - infra/app_train_loop.py::train_loop_smoke  (parent H-GRPO + producer)
  - infra/app_train_turnrd.py::train_turnrd_run  (standalone TurnRD fit)

Both already accept the `--config` / `--replay` / `--ckpt-out` flags
this script needs.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "method_hgpo_turnrd.json"


@dataclass
class OrchestrationConfig:
    """All knobs the orchestration loop needs.

    Defaults match a 200-episode protocol run split into 5 rounds of 40
    episodes each, with 3 epochs of TurnRD fitting between rounds.
    """

    config_path: Path = DEFAULT_CONFIG
    rounds: int = 5
    episodes_per_round: int = 40
    turnrd_epochs: int = 3
    turnrd_mode: int = 1
    turnrd_batch_size: int = 16
    turnrd_lr: float = 1e-4
    # Modal volume paths — must match `cfg.turnrd.replay_buffer_path`
    # and `cfg.turnrd.ckpt_path` in the JSON config (we cross-check
    # below to surface mismatches before launching anything on Modal).
    replay_path: str = "/vol/cache/turnrd_replay.jsonl"
    ckpt_path: str = "/vol/cache/turnrd_ckpt.pt"
    run_name_prefix: str = "method_b_orchestrated"
    # Optional SFT-warm-started LoRA adapter on the Modal volume. When
    # set, the parent train_loop loads it via PEFT.load_adapter() and
    # syncs the merged weights into vLLM before the first episode.
    # Without this, cold-start RL on real WebShop typically produces
    # R≈0 for hundreds of episodes — the protocol's TurnRD signal
    # depends on having non-trivially-trained policy producing
    # reward variance from episode 0.
    sft_adapter: str = ""
    # Held-out eval pass appended to each train_loop call. Uses
    # greedy sampling on a disjoint task range so the eval is stable
    # AND comparable across rounds + methods + seeds. Default
    # `[6500, 6550)` is INSIDE WebShop's ~6910-goal range AND disjoint
    # from training task ranges (seed 11 → [2200, 2400),
    # seed 23 → [4600, 4800)). Higher offsets like 10000 raise
    # `IndexError` in WebShop's `web_agent_text_env.py:512`. Set
    # --eval-episodes 0 to disable.
    eval_episodes: int = 50
    eval_task_id_base: int = 6500
    # Multi-seed protocol support. None ⇒ no seed-specific offset
    # applied (legacy single-run behavior). When set, each seed gets a
    # disjoint task_id range so different seeds never train on the same
    # WebShop tasks. Also tags the run-name-prefix with `_seed{N}`.
    seed: int | None = None
    # Env-name dispatch (default `webshop` for backward compat with
    # all prior single-env sweeps). When set to `alfworld`, the
    # orchestrator calls `train_loop_alfworld.remote(...)` (binding
    # the AlfWorld-runtime image) and the eval-task-id-range guard is
    # widened — AlfWorld's adapter wraps task_id with `% len(games)`
    # so any non-negative integer is safe, but training/eval ranges
    # must still be disjoint per the same per-seed offset math.
    env_name: str = "webshop"
    dry_run: bool = False
    skip_warmup_fit: bool = False
    extra_train_loop_args: list[str] = field(default_factory=list)
    # Multi-round protocol with policy carry-across:
    # - When False (default), every round loads `cfg.sft_adapter` so the
    #   policy resets to the SFT warm-start each round (legacy behavior).
    # - When True, round 0 loads `sft_adapter`, but rounds N>=1 load the
    #   adapter saved by round N-1 (path = `<adapter_dir>/<run_prefix>_round{N-1:02d}_adapter`).
    #   The corresponding `--save-adapter-out` is added to every round's
    #   train_loop call so each round persists its trained LoRA adapter.
    #   This makes 8x40 ep act like 1x320 ep (continuous training)
    #   instead of 8 independent shots from SFT.
    carry_policy_across_rounds: bool = False
    adapter_dir: str = "/vol/checkpoints"
    extra_turnrd_args: list[str] = field(default_factory=list)

    @property
    def base_task_id_offset(self) -> int:
        """Per-seed deterministic task_id starting offset.

        Each seed gets a slice of length `rounds * episodes_per_round`
        starting at `seed * rounds * episodes_per_round`. Two seeds
        are guaranteed to operate on disjoint task ranges as long as
        the per-run cap stays the same.

        When `seed is None`, returns 0 (legacy behavior — preserves
        the single-run case where the caller may want to manually
        thread a `--task-id-offset` through `--extra-train-loop-args`).
        """
        if self.seed is None:
            return 0
        return int(self.seed) * int(self.rounds) * int(self.episodes_per_round)

    @property
    def effective_run_name_prefix(self) -> str:
        """`run_name_prefix` with the seed appended when set."""
        if self.seed is None:
            return self.run_name_prefix
        return f"{self.run_name_prefix}_seed{int(self.seed)}"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> OrchestrationConfig:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Held-out eval episodes appended to each train_loop call "
             "(greedy sampling, K=1, disjoint task range). Set to 0 to "
             "disable. Default 50.",
    )
    parser.add_argument(
        "--eval-task-id-base",
        type=int,
        default=6500,
        help="Starting task ID for the held-out eval range. The range "
             "[base, base+eval_episodes) MUST be (a) WITHIN WebShop's "
             "~6910-goal limit (default num_products=1000) and (b) "
             "disjoint from training task IDs (see --seed's "
             "base_task_id_offset). Default 6500.",
    )
    parser.add_argument(
        "--sft-adapter",
        default="",
        help="Optional SFT-warm-started LoRA adapter path on the Modal "
             "volume (e.g. /vol/checkpoints/sft_v3_<ts>). Forwarded to "
             "the parent train_loop as --sft-adapter so the policy "
             "starts from a non-trivially-trained checkpoint. Without "
             "this, cold-start RL on real WebShop typically produces "
             "R≈0 for hundreds of episodes.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the H-GRPO method config (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of producer↔trainer alternations (default: 5).",
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=40,
        help="H-GRPO episodes per round (default: 40 → 5 × 40 = 200 total).",
    )
    parser.add_argument(
        "--turnrd-epochs",
        type=int,
        default=3,
        help="Standalone TurnRD epochs between rounds (default: 3).",
    )
    parser.add_argument(
        "--turnrd-mode",
        type=int,
        choices=(1, 2),
        default=1,
        help="TurnRD training mode: 1=predict R, 2=distill judge labels (default: 1).",
    )
    parser.add_argument(
        "--turnrd-batch-size",
        type=int,
        default=16,
        help="Standalone TurnRD batch size (default: 16).",
    )
    parser.add_argument(
        "--turnrd-lr",
        type=float,
        default=1e-4,
        help="Standalone TurnRD learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--replay-path",
        default="/vol/cache/turnrd_replay.jsonl",
        help="Modal-volume replay JSONL path; must match cfg.turnrd.replay_buffer_path.",
    )
    parser.add_argument(
        "--ckpt-path",
        default="/vol/cache/turnrd_ckpt.pt",
        help="Modal-volume TurnRD ckpt path; must match cfg.turnrd.ckpt_path.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default="method_b_orchestrated",
        help="Prefix for per-round run names (default: method_b_orchestrated). "
             "When --seed is set, '_seed{N}' is appended.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional protocol seed. When set, drives a deterministic "
             "task_id_offset slice (seed * rounds * episodes_per_round) so "
             "different seeds train on disjoint WebShop tasks, AND tags the "
             "run-name-prefix with '_seed{N}'. Required for the multi-seed "
             "protocol sweep; omit for ad-hoc single-run launches.",
    )
    parser.add_argument(
        "--env-name",
        choices=("webshop", "alfworld"),
        default="webshop",
        help="Which env's train_loop entrypoint to invoke. Default "
             "'webshop' preserves backward compat with all prior sweeps. "
             "'alfworld' routes to `train_loop_alfworld` (the lighter "
             "alfworld_image, no Java/pyserini/spaCy) and widens the "
             "eval-task-id-range pre-flight check.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the modal commands without executing them.",
    )
    parser.add_argument(
        "--skip-warmup-fit",
        action="store_true",
        help="Skip the TurnRD fit at the end of round 0 (useful when you "
             "have a pre-fit ckpt staged at --ckpt-path).",
    )
    parser.add_argument(
        "--carry-policy-across-rounds",
        action="store_true",
        help="When set, round 0 loads --sft-adapter; rounds N>=1 load the "
             "LoRA adapter saved by round N-1 (path = "
             "<adapter-dir>/<run-prefix>_round<N-1:02d>_adapter). Each "
             "round's train_loop also gets a --save-adapter-out flag so it "
             "persists its trained adapter for the next round to pick up. "
             "This makes the multi-round protocol behave like one long "
             "continuous training run instead of N independent shots from "
             "SFT (which is the legacy behavior, preserved by default).",
    )
    parser.add_argument(
        "--adapter-dir",
        default="/vol/checkpoints",
        help="Directory on the Modal volume where per-round saved LoRA "
             "adapters are written (only used when --carry-policy-across-rounds "
             "is set). Each round writes "
             "<adapter-dir>/<run-prefix>_round<N:02d>_adapter/. "
             "Default '/vol/checkpoints'.",
    )
    parser.add_argument(
        "--extra-train-loop-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra positional args forwarded to `modal run app_train_loop.py` (after --).",
    )
    args = parser.parse_args(list(argv))

    return OrchestrationConfig(
        config_path=args.config,
        rounds=args.rounds,
        episodes_per_round=args.episodes_per_round,
        turnrd_epochs=args.turnrd_epochs,
        turnrd_mode=args.turnrd_mode,
        turnrd_batch_size=args.turnrd_batch_size,
        turnrd_lr=args.turnrd_lr,
        replay_path=args.replay_path,
        ckpt_path=args.ckpt_path,
        run_name_prefix=args.run_name_prefix,
        sft_adapter=args.sft_adapter,
        eval_episodes=args.eval_episodes,
        eval_task_id_base=args.eval_task_id_base,
        seed=args.seed,
        env_name=args.env_name,
        dry_run=args.dry_run,
        skip_warmup_fit=args.skip_warmup_fit,
        extra_train_loop_args=list(args.extra_train_loop_args or []),
        carry_policy_across_rounds=bool(args.carry_policy_across_rounds),
        adapter_dir=str(args.adapter_dir),
    )


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------


def _preflight(cfg: OrchestrationConfig) -> None:
    """Catch obvious mistakes before spending money on Modal.

    1. Modal CLI is on PATH.
    2. Config file exists and is parseable JSON.
    3. Config's `turnrd.replay_buffer_path` and `turnrd.ckpt_path` match
       the orchestration's `--replay-path` and `--ckpt-path` (otherwise
       the producer writes to one path and the standalone trainer reads
       from another — a silent split-brain).
    4. Round count > 0.
    """
    if shutil.which("modal") is None and not cfg.dry_run:
        raise SystemExit(
            "ERROR: `modal` CLI not found on PATH. Install with `pip install modal`, "
            "authenticate (`modal token new`), then re-run. (Use --dry-run to skip.)"
        )

    if not cfg.config_path.is_file():
        raise SystemExit(f"ERROR: config file not found: {cfg.config_path}")
    try:
        with open(cfg.config_path) as fh:
            cfg_json = json.load(fh)
    except json.JSONDecodeError as e:
        raise SystemExit(
            f"ERROR: {cfg.config_path} is not valid JSON: {e}"
        ) from e

    turnrd_cfg = cfg_json.get("turnrd", {}) if isinstance(cfg_json, dict) else {}
    if not isinstance(turnrd_cfg, dict):
        raise SystemExit(
            f"ERROR: {cfg.config_path} has no top-level 'turnrd' block; "
            "this orchestration only makes sense for Method-B configs."
        )
    decomposer = (cfg_json.get("hgpo", {}) or {}).get("decomposer")
    if decomposer != "turnrd":
        raise SystemExit(
            f"ERROR: {cfg.config_path}::hgpo.decomposer = {decomposer!r}, "
            "expected 'turnrd'. This orchestrator coordinates "
            "the TurnRD producer + standalone trainer; non-TurnRD "
            "configs should run directly via "
            "`modal run infra/app_train_loop.py --config ...`."
        )

    cfg_replay = turnrd_cfg.get("replay_buffer_path")
    if cfg_replay and cfg_replay != cfg.replay_path:
        raise SystemExit(
            f"ERROR: orchestration --replay-path={cfg.replay_path!r} does not "
            f"match {cfg.config_path}::turnrd.replay_buffer_path={cfg_replay!r}. "
            "Producer would write to one file and the standalone trainer "
            "would read another — a silent split-brain. Align them before launching."
        )
    cfg_ckpt = turnrd_cfg.get("ckpt_path")
    if cfg_ckpt and cfg_ckpt != cfg.ckpt_path:
        raise SystemExit(
            f"ERROR: orchestration --ckpt-path={cfg.ckpt_path!r} does not "
            f"match {cfg.config_path}::turnrd.ckpt_path={cfg_ckpt!r}. "
            "Standalone trainer would write a ckpt the parent's refresh "
            "fn never reads. Align them before launching."
        )

    # Architecture version consistency: the parent train_loop and the
    # standalone fitter MUST construct the SAME `TurnRD` vs `TurnRDv2`
    # class for the refresh-fn ckpt load to succeed (state_dict keys
    # differ across architectures). Both sides read `version` from the
    # JSON; we just defensively reject obviously-wrong values here so
    # a typo doesn't silently flip back to v1 default.
    cfg_version = turnrd_cfg.get("version", "v1")
    if str(cfg_version).lower() not in ("v1", "v2"):
        raise SystemExit(
            f"ERROR: {cfg.config_path}::turnrd.version = {cfg_version!r}; "
            "expected 'v1' or 'v2'. State-dict load between rounds "
            "would silently break with a typo here."
        )

    if cfg.rounds <= 0:
        raise SystemExit(f"ERROR: --rounds must be positive; got {cfg.rounds}.")
    if cfg.episodes_per_round <= 0:
        raise SystemExit(
            f"ERROR: --episodes-per-round must be positive; got {cfg.episodes_per_round}."
        )

    # Env-aware eval-task-id range guard. WebShop's `web_agent_text_env.py`
    # holds a finite `goals` list (~6910 with default `num_products=1000`);
    # `eval_task_id_base + eval_episodes` must be within that range. AlfWorld
    # has a much smaller eval pool (~140 valid_seen + ~140 valid_unseen) but
    # the adapter wraps task_id with `% len(games)`, so any non-negative
    # integer is technically safe. We still warn when the requested range
    # overlaps the per-seed training slice, since training/eval should be
    # disjoint regardless of env.
    if cfg.env_name == "webshop":
        WEBSHOP_GOALS_LEN = 6910  # at default num_products=1000
        if cfg.eval_task_id_base + cfg.eval_episodes > WEBSHOP_GOALS_LEN:
            raise SystemExit(
                f"ERROR: WebShop eval range "
                f"[{cfg.eval_task_id_base}, "
                f"{cfg.eval_task_id_base + cfg.eval_episodes}) exceeds the "
                f"WebShop goals pool (~{WEBSHOP_GOALS_LEN}). Lower "
                "--eval-task-id-base or --eval-episodes."
            )
    train_lo = cfg.base_task_id_offset
    train_hi = cfg.base_task_id_offset + cfg.rounds * cfg.episodes_per_round
    eval_lo = cfg.eval_task_id_base
    eval_hi = cfg.eval_task_id_base + cfg.eval_episodes
    if not (eval_hi <= train_lo or eval_lo >= train_hi):
        raise SystemExit(
            f"ERROR: eval task range [{eval_lo}, {eval_hi}) overlaps "
            f"training task range [{train_lo}, {train_hi}). Pick a "
            "disjoint --eval-task-id-base."
        )


# ---------------------------------------------------------------------------
# Modal command builders
# ---------------------------------------------------------------------------


def _to_container_path(local_path: Path) -> str:
    """Translate a local config path to its mounted location inside Modal.

    `infra/image.py::_add_workspace` mounts the repo root at `/workspace`
    inside the container. The orchestrator runs on the host with absolute
    paths under `REPO_ROOT`; passing those raw to `modal run` would land
    `open("/Users/.../config.json")` inside a container where that path
    doesn't exist. Translate to `/workspace/<relative-to-repo>` so the
    container reads from the mounted source tree.
    """
    abs_path = local_path.resolve()
    try:
        rel = abs_path.relative_to(REPO_ROOT)
    except ValueError as e:
        raise SystemExit(
            f"ERROR: config {abs_path} is outside REPO_ROOT={REPO_ROOT}; "
            "cannot map to a Modal /workspace path. Place the config "
            "inside the repo before launching."
        ) from e
    return f"/workspace/{rel.as_posix()}"


_APP_ID_RE = re.compile(r"ap-[A-Za-z0-9]+")


def _parse_app_id(modal_run_stdout: str) -> str | None:
    """Extract the ephemeral app ID from `modal run --detach` output.

    Modal prints e.g. `View run at https://modal.com/apps/shoupei/main/ap-...`
    on stdout. We grep for the first `ap-XXXXX` token.
    """
    m = _APP_ID_RE.search(modal_run_stdout)
    return m.group(0) if m else None


def _wait_for_app_finish(
    app_id: str, *, label: str, poll_interval_s: float = 5.0,
    timeout_s: float = 60 * 60,
) -> int:
    """Poll `modal app list` until `app_id` leaves the running state.

    Returns 0 if the app finished cleanly, non-zero on timeout / unknown
    state. Streams a heartbeat every `poll_interval_s` seconds so the
    user sees progress.
    """
    print(f"   ↻ polling {app_id} ({label})…")
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"   ⚠ timeout waiting for {app_id} after {elapsed:.0f}s.")
            return 124
        try:
            res = subprocess.run(
                ["modal", "app", "list"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            print("   ⚠ `modal app list` timed out; retrying.")
            time.sleep(poll_interval_s)
            continue
        if res.returncode != 0:
            print(f"   ⚠ `modal app list` exited {res.returncode}; retrying.")
            time.sleep(poll_interval_s)
            continue
        # Find our row.
        state = None
        for line in res.stdout.splitlines():
            if app_id in line:
                # Heuristic: strip ANSI/box-drawing chars and look for
                # known states. State is one of: ephemeral, stopped,
                # stopping..., (running labels we treat as not-done).
                if "stopped" in line:
                    state = "stopped"
                elif "stopping" in line:
                    state = "stopping"
                elif "ephemeral" in line:
                    state = "ephemeral"
                else:
                    state = "unknown"
                break
        if state == "stopped":
            print(f"   ✓ {app_id} finished after {elapsed:.0f}s.")
            return 0
        if state is None:
            # Newly-submitted apps may not appear in `app list` for a
            # few seconds — keep waiting rather than fail.
            print(
                f"   ↻ {app_id} not yet visible in app list (elapsed: {elapsed:.0f}s)…"
            )
        else:
            print(
                f"   ↻ {app_id} still {state} (elapsed: {elapsed:.0f}s)…"
            )
        time.sleep(poll_interval_s)


def _train_loop_cmd(cfg: OrchestrationConfig, round_idx: int) -> list[str]:
    """`modal run --detach infra/app_train_loop.py --config <cfg> --n-episodes M ...`

    The `app_train_loop.py::main()` entrypoint accepts:
      --n-episodes / --k / --max-turns / --task-id-offset / --num-products /
      --sync-every / --run-name / --sft-adapter / --use-sft-as-ref /
      --kl-warmup-episodes / --gpu-mem-util / --config

    We rely on the JSON config to set most of these via the `--config`
    switch; only `--n-episodes`, `--k`, `--task-id-offset`,
    and `--run-name` are overridden round-by-round (and seed-by-seed).
    `--k` is read from `cfg.config_path` JSON's
    `train.K_trajectories_per_task` so the protocol K matches what the
    user configured (the JSON-driven app no longer overrides this on
    its own).

    `--config` is translated to its in-container `/workspace/...` path
    so `open(...)` inside the Modal function actually finds the file.

    `--detach` is passed to `modal run` so the cloud function keeps
    running even if the local orchestrator process dies. The
    orchestrator polls for completion via `modal app list` between
    rounds (see `_wait_for_app_finish`).
    """
    # Read K from the JSON; default to the app's own default (4) if
    # the key is absent.
    try:
        with open(cfg.config_path) as fh:
            cfg_json = json.load(fh)
    except Exception:
        cfg_json = {}
    k_per_task = int(
        (cfg_json.get("train", {}) or {}).get("K_trajectories_per_task", 4)
    )

    cmd = [
        "modal", "run", "--detach",
        f"infra/app_train_loop.py::train_loop_{cfg.env_name}",
        "--config", _to_container_path(cfg.config_path),
        "--n-episodes", str(cfg.episodes_per_round),
        "--k", str(k_per_task),
        "--task-id-offset", str(
            cfg.base_task_id_offset + round_idx * cfg.episodes_per_round
        ),
        "--run-name", f"{cfg.effective_run_name_prefix}_round{round_idx:02d}",
        "--round-idx", str(round_idx),
    ]
    # `gpu_mem_util` from the JSON's train block, when present. This
    # caps vLLM's KV cache so the trainer has enough activation room
    # for grad-tracking forward passes (OOM mitigation).
    gpu_mem_util_cfg = (cfg_json.get("train", {}) or {}).get("gpu_mem_util")
    if gpu_mem_util_cfg is not None:
        cmd.extend(["--gpu-mem-util", str(float(gpu_mem_util_cfg))])
    # Adapter routing:
    # - Legacy (carry_policy_across_rounds=False): every round loads the
    #   same `cfg.sft_adapter`, resetting the policy to SFT each round.
    # - Carry-policy mode (carry_policy_across_rounds=True): round 0
    #   loads `cfg.sft_adapter`; rounds N>=1 load the previous round's
    #   saved adapter so training accumulates across rounds. Each round
    #   also writes its trained adapter to `--save-adapter-out` for the
    #   NEXT round to consume.
    if cfg.carry_policy_across_rounds:
        if round_idx == 0:
            load_adapter = cfg.sft_adapter
        else:
            load_adapter = (
                f"{cfg.adapter_dir.rstrip('/')}/"
                f"{cfg.effective_run_name_prefix}_round{round_idx - 1:02d}_adapter"
            )
        save_adapter_out = (
            f"{cfg.adapter_dir.rstrip('/')}/"
            f"{cfg.effective_run_name_prefix}_round{round_idx:02d}_adapter"
        )
        if load_adapter:
            cmd.extend(["--sft-adapter", load_adapter])
        cmd.extend(["--save-adapter-out", save_adapter_out])
    elif cfg.sft_adapter:
        cmd.extend(["--sft-adapter", cfg.sft_adapter])
    cmd.extend(["--eval-episodes", str(cfg.eval_episodes)])
    cmd.extend(["--eval-task-id-base", str(cfg.eval_task_id_base)])
    cmd.extend(cfg.extra_train_loop_args)
    return cmd


def _train_turnrd_cmd(cfg: OrchestrationConfig) -> list[str]:
    """`modal run --detach infra/app_train_turnrd.py --replay <p> --mode N ...`

    Round-independent: every round reads and writes the same shared
    paths on the Modal Volume. The replay file accumulates across
    rounds; the ckpt is overwritten so the parent's refresh_fn always
    loads the freshest fit.

    Detached for the same reason as `_train_loop_cmd`.

    Forwards aux-loss knobs (`lambda_value`, `gamma`, `lambda_entropy`)
    + architecture knobs (`causal`, `value_head`, larger `layers`/
    `hidden_size`) from the JSON config so the standalone trainer
    matches what the parent's TurnRD model was built with. Without
    these, the standalone trainer would fall back to its own defaults
    and produce an architecturally-mismatched ckpt.
    """
    # Read TurnRD knobs from the JSON; fall back to standalone-trainer defaults.
    try:
        with open(cfg.config_path) as fh:
            cfg_json = json.load(fh)
    except Exception:
        cfg_json = {}
    turnrd_block = (cfg_json.get("turnrd", {}) or {})

    # Resolve turnrd_lr: prefer JSON's value, fall back to CLI cfg default.
    effective_turnrd_lr = float(turnrd_block.get("turnrd_lr", cfg.turnrd_lr))

    cmd = [
        "modal", "run", "--detach", "infra/app_train_turnrd.py",
        "--replay", cfg.replay_path,
        "--mode", str(cfg.turnrd_mode),
        "--n-epochs", str(cfg.turnrd_epochs),
        "--batch-size", str(cfg.turnrd_batch_size),
        "--lr", str(effective_turnrd_lr),
        "--ckpt-out", cfg.ckpt_path,
    ]
    # Architecture: must match what the parent train_loop's
    # build_trainer_from_config built, otherwise refresh-fn ckpt load
    # will silently break (state_dict keys mismatch).
    for cli, jkey, default in [
        ("--version", "version", None),
        ("--layers", "layers", None),
        ("--hidden-size", "hidden_size", None),
        ("--n-heads", "n_heads", None),
        ("--max-turns", "max_turns", None),
        ("--dropout", "dropout", None),
        ("--progress-prior-strength", "progress_prior_strength", None),
        ("--lambda-value", "lambda_value", None),
        ("--gamma", "gamma", None),
        ("--lambda-entropy", "lambda_entropy", None),
        ("--lambda-contrastive", "lambda_contrastive", None),
        ("--contrastive-temperature", "contrastive_temperature", None),
        ("--lambda-rank", "lambda_rank", None),
        ("--lambda-progress", "lambda_progress", None),
        ("--rank-margin", "rank_margin", None),
    ]:
        if jkey in turnrd_block:
            cmd.extend([cli, str(turnrd_block[jkey])])
    # Boolean flags use Click/Modal convention: `--flag` (true) /
    # `--no-flag` (false), NOT `--flag value`. Translate the JSON's
    # bool to the right form.
    def _bool_flag(name: str, value: bool) -> list[str]:
        return [f"--{name}" if value else f"--no-{name}"]

    if "causal" in turnrd_block:
        cmd.extend(_bool_flag("causal", bool(turnrd_block["causal"])))
    if "value_head" in turnrd_block:
        cmd.extend(_bool_flag("value-head", bool(turnrd_block["value_head"])))
    cmd.extend(cfg.extra_turnrd_args)
    return cmd


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, dry_run: bool, label: str) -> int:
    """Submit a `modal run --detach …` command, parse its app ID, then
    poll `modal app list` until that app finishes.

    Returns 0 on success, non-zero on failure.

    Two-phase shape: we want each cloud function to keep running even
    if this orchestrator process dies, so each `modal run` is detached
    (the local CLI returns immediately after submitting). We then
    wait on the cloud-side state via `_wait_for_app_finish`, which
    only requires the local CLI to be alive *for polling* (not for
    the actual compute). If the orchestrator dies mid-poll, the cloud
    job continues; `modal app logs <app_id>` can be used to recover.

    Robustness: a non-zero CLI exit DOES NOT mean the cloud function
    failed. The Modal CLI can exit non-zero on transient
    upload/download glitches (e.g. DNS hiccup at the very end of a
    long run) even though the cloud function completed successfully.
    Whenever we can parse an app ID from the CLI output, we trust the
    CLOUD STATE (via `_wait_for_app_finish`) rather than the LOCAL
    CLI exit code. Only a missing app ID is treated as a hard
    submission failure.
    """
    print(f"\n┌── {label}")
    print(f"│  $ {' '.join(cmd)}")
    print("└──")
    if dry_run:
        return 0
    t0 = time.time()
    # Phase 1: submit detached. Capture stdout so we can parse the app ID.
    submit = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True
    )
    print(submit.stdout)
    if submit.stderr:
        print(submit.stderr, file=sys.stderr)
    # Try to parse the app ID from EITHER stream regardless of exit code.
    app_id = _parse_app_id(submit.stdout) or _parse_app_id(submit.stderr)
    if app_id is None:
        # No app ID at all → the submit truly failed before reaching Modal.
        print(
            f"⚠ Could not parse app ID from `modal run --detach` output. "
            f"The submit likely failed before the cloud function started "
            f"(CLI exit={submit.returncode}). Aborting this round."
        )
        return submit.returncode if submit.returncode != 0 else 1
    if submit.returncode != 0:
        # CLI exited non-zero but we DO have an app ID. Trust the cloud
        # state instead — Modal CLI's exit code is unreliable on
        # transient network glitches.
        print(
            f"⚠ Modal CLI exited {submit.returncode} but app {app_id} was "
            f"submitted. Trusting cloud state via polling."
        )
    # Phase 2: poll for completion.
    rc = _wait_for_app_finish(app_id, label=label)
    elapsed = round(time.time() - t0, 2)
    print(f"({label} exited {rc} after {elapsed}s)")
    return rc


def _orchestrate(cfg: OrchestrationConfig) -> int:
    """Execute the round-robin loop. Returns the final exit code (0 on success)."""
    print("=== Method-B (TurnRD) end-to-end orchestration ===")
    print(f"  env name           : {cfg.env_name}")
    print(f"  config             : {cfg.config_path}")
    print(f"  rounds             : {cfg.rounds}")
    print(f"  episodes/round     : {cfg.episodes_per_round}")
    print(f"  turnrd epochs/round: {cfg.turnrd_epochs}")
    print(f"  turnrd mode        : {cfg.turnrd_mode}")
    print(f"  replay path (vol)  : {cfg.replay_path}")
    print(f"  ckpt   path (vol)  : {cfg.ckpt_path}")
    print(f"  run name prefix    : {cfg.effective_run_name_prefix}")
    print(f"  seed               : {cfg.seed if cfg.seed is not None else '(none)'}")
    print(f"  base task offset   : {cfg.base_task_id_offset}")
    print(f"  dry-run            : {cfg.dry_run}")
    print(f"  skip warmup fit    : {cfg.skip_warmup_fit}")

    for round_idx in range(cfg.rounds):
        # ---- (a) Parent H-GRPO loop: trains policy + emits replay rows.
        rc = _run(
            _train_loop_cmd(cfg, round_idx),
            dry_run=cfg.dry_run,
            label=f"Round {round_idx}: train_loop ({cfg.episodes_per_round} eps)",
        )
        if rc != 0:
            print(
                f"ERROR: train_loop in round {round_idx} exited {rc}. Aborting "
                "orchestration; replay buffer state is preserved on the volume "
                "for inspection."
            )
            return rc

        # ---- (b) Standalone TurnRD fit on the accumulated replay buffer.
        if round_idx == 0 and cfg.skip_warmup_fit:
            print(
                f"Round {round_idx}: skipping standalone TurnRD fit "
                "(--skip-warmup-fit). Next round's refresh_fn will read "
                f"whatever ckpt is at {cfg.ckpt_path}."
            )
            continue
        rc = _run(
            _train_turnrd_cmd(cfg),
            dry_run=cfg.dry_run,
            label=f"Round {round_idx}: train_turnrd ({cfg.turnrd_epochs} epochs)",
        )
        if rc != 0:
            print(
                f"ERROR: train_turnrd in round {round_idx} exited {rc}. Aborting; "
                "ckpt may be stale or absent for next round."
            )
            return rc

    print(
        f"\n=== Done. {cfg.rounds} rounds × {cfg.episodes_per_round} episodes = "
        f"{cfg.rounds * cfg.episodes_per_round} total H-GRPO episodes. ==="
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])
    _preflight(cfg)
    return _orchestrate(cfg)


if __name__ == "__main__":
    sys.exit(main())
