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

Requires `modal` on PATH. The two Modal app entrypoints invoked are:
  - infra/app_train_loop.py::train_loop_smoke  (parent H-GRPO + producer)
  - infra/app_train_turnrd.py::train_turnrd_run  (standalone TurnRD fit)

Both already accept the `--config` / `--replay` / `--ckpt-out` flags
this script needs.
"""
from __future__ import annotations

import argparse
import json
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
    # Multi-seed protocol support. None ⇒ no seed-specific offset
    # applied (legacy single-run behavior). When set, each seed gets a
    # disjoint task_id range so different seeds never train on the same
    # WebShop tasks. Also tags the run-name-prefix with `_seed{N}`.
    seed: int | None = None
    dry_run: bool = False
    skip_warmup_fit: bool = False
    extra_train_loop_args: list[str] = field(default_factory=list)
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
        seed=args.seed,
        dry_run=args.dry_run,
        skip_warmup_fit=args.skip_warmup_fit,
        extra_train_loop_args=list(args.extra_train_loop_args or []),
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
            "expected 'turnrd'. This orchestrator coordinates the TurnRD "
            "producer + standalone trainer; non-TurnRD configs should run "
            "directly via `modal run infra/app_train_loop.py --config ...`."
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

    if cfg.rounds <= 0:
        raise SystemExit(f"ERROR: --rounds must be positive; got {cfg.rounds}.")
    if cfg.episodes_per_round <= 0:
        raise SystemExit(
            f"ERROR: --episodes-per-round must be positive; got {cfg.episodes_per_round}."
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


def _train_loop_cmd(cfg: OrchestrationConfig, round_idx: int) -> list[str]:
    """`modal run infra/app_train_loop.py --config <cfg> --n-episodes M ...`

    The `app_train_loop.py::main()` entrypoint accepts:
      --n-episodes / --k / --max-turns / --task-id-offset / --num-products /
      --sync-every / --run-name / --sft-adapter / --use-sft-as-ref /
      --kl-warmup-episodes / --gpu-mem-util / --config

    We rely on the JSON config to set most of these (per the Day-14
    `--config` switch); only `--n-episodes`, `--k`, `--task-id-offset`,
    and `--run-name` are overridden round-by-round (and seed-by-seed).
    `--k` is read from `cfg.config_path` JSON's
    `train.K_trajectories_per_task` so the protocol K matches what the
    user configured (the JSON-driven app no longer overrides this on
    its own — see the post-mortem in the execution plan).

    `--config` is translated to its in-container `/workspace/...` path
    so `open(...)` inside the Modal function actually finds the file.
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

    return [
        "modal", "run", "infra/app_train_loop.py",
        "--config", _to_container_path(cfg.config_path),
        "--n-episodes", str(cfg.episodes_per_round),
        "--k", str(k_per_task),
        "--task-id-offset", str(
            cfg.base_task_id_offset + round_idx * cfg.episodes_per_round
        ),
        "--run-name", f"{cfg.effective_run_name_prefix}_round{round_idx:02d}",
        *cfg.extra_train_loop_args,
    ]


def _train_turnrd_cmd(cfg: OrchestrationConfig) -> list[str]:
    """`modal run infra/app_train_turnrd.py --replay <p> --mode N ...`

    Round-independent: every round reads and writes the same shared
    paths on the Modal Volume. The replay file accumulates across
    rounds; the ckpt is overwritten so the parent's refresh_fn always
    loads the freshest fit.
    """
    return [
        "modal", "run", "infra/app_train_turnrd.py",
        "--replay", cfg.replay_path,
        "--mode", str(cfg.turnrd_mode),
        "--n-epochs", str(cfg.turnrd_epochs),
        "--batch-size", str(cfg.turnrd_batch_size),
        "--lr", str(cfg.turnrd_lr),
        "--ckpt-out", cfg.ckpt_path,
        *cfg.extra_turnrd_args,
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, dry_run: bool, label: str) -> int:
    """Run a `modal run ...` command, streaming its output. Returns exit code."""
    print(f"\n┌── {label}")
    print(f"│  $ {' '.join(cmd)}")
    print("└──")
    if dry_run:
        return 0
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = round(time.time() - t0, 2)
    print(f"({label} exited {proc.returncode} after {elapsed}s)")
    return proc.returncode


def _orchestrate(cfg: OrchestrationConfig) -> int:
    """Execute the round-robin loop. Returns the final exit code (0 on success)."""
    print("=== Method-B (TurnRD) end-to-end orchestration ===")
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
