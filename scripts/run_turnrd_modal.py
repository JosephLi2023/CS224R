#!/usr/bin/env python3
"""Method-B (TurnRD) end-to-end orchestration on Modal.

Round-robins the producer (parent H-GRPO train_loop) and the standalone TurnRD
trainer so the policy keeps training while TurnRD's reward decomposition is
periodically re-fit from the accumulated replay buffer. /vol/... paths are
shared across Modal containers via the project Volume.

Usage:
  scripts/run_turnrd_modal.py --rounds 5 --episodes-per-round 40
  scripts/run_turnrd_modal.py --dry-run    # print commands only

Requires `modal` on PATH. For long unattended runs, launch under nohup so the
orchestration isn't bounded by an IDE/agent session timeout.
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

    Defaults: a 200-episode run split into 5 rounds of 40 episodes, with 3
    epochs of TurnRD fitting between rounds.
    """

    config_path: Path = DEFAULT_CONFIG
    rounds: int = 5
    start_round: int = 0
    episodes_per_round: int = 40
    turnrd_epochs: int = 3
    turnrd_mode: int = 1
    turnrd_batch_size: int = 16
    turnrd_lr: float = 1e-4
    # Modal volume paths; must match cfg.turnrd.replay_buffer_path / ckpt_path
    # (cross-checked in _preflight).
    replay_path: str = "/vol/cache/turnrd_replay.jsonl"
    ckpt_path: str = "/vol/cache/turnrd_ckpt.pt"
    run_name_prefix: str = "method_b_orchestrated"
    # Optional SFT-warm-started LoRA adapter. Without it, cold-start RL on real
    # WebShop typically produces R~0 for a long time.
    sft_adapter: str = ""
    # Held-out eval pass appended to each train_loop call (greedy, disjoint
    # task range). Set --eval-episodes 0 to disable.
    eval_episodes: int = 50
    eval_task_id_base: int = 6500
    # Multi-seed protocol. None -> no seed offset (legacy single-run). When set,
    # each seed gets a disjoint task_id range and a _seed{N} run-name suffix.
    seed: int | None = None
    # Env dispatch: 'webshop' (default) or 'alfworld' (routes to
    # train_loop_alfworld and widens the eval-task-id guard).
    env_name: str = "webshop"
    dry_run: bool = False
    skip_warmup_fit: bool = False
    extra_train_loop_args: list[str] = field(default_factory=list)
    # Carry-policy across rounds: when True, R0 loads cfg.sft_adapter and R_N>0
    # loads R_{N-1}'s saved adapter (continuous training); each round adds
    # --save-adapter-out. Default False resets to SFT each round.
    carry_policy_across_rounds: bool = False
    adapter_dir: str = "/vol/checkpoints"
    # vLLM rollout temperature for K-trajectory training (eval stays greedy).
    # Lower values (e.g. 0.7) reduce mode-collapse / dead-K on saturated policies.
    rollout_temperature: float = 1.0
    # Cross-run lineage opt-in: with --carry-policy-across-rounds, the first
    # iterated round loads cfg.sft_adapter instead of the prefix-derived path.
    # Default False keeps the legacy resume convention.
    sft_adapter_overrides_derived: bool = False
    extra_turnrd_args: list[str] = field(default_factory=list)

    @property
    def base_task_id_offset(self) -> int:
        """Per-seed task_id offset = seed * rounds * episodes_per_round.

        Guarantees disjoint task ranges across seeds. Returns 0 when seed is
        None (legacy single-run behavior).
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


# Argument parsing


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
             "R~0 for hundreds of episodes.",
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
        help="Number of producer<->trainer alternations (default: 5).",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        default=0,
        help="Round index to START from (default: 0). Use to resume a "
             "partial run after a Modal timeout. Round adapter chain "
             "expects /vol/checkpoints/<run-prefix>_round{start-round-1:02d}_adapter "
             "to exist when --carry-policy-across-rounds is set and "
             "start-round >= 1; use that path as --sft-adapter to "
             "preserve adapter continuity from the prior run.",
    )
    parser.add_argument(
        "--episodes-per-round",
        type=int,
        default=40,
        help="H-GRPO episodes per round (default: 40 -> 5 x 40 = 200 total).",
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
        "--rollout-temperature",
        type=float,
        default=1.0,
        help="vLLM SamplingParams temperature for the K-trajectory training "
             "rollouts (eval stays greedy at T=0.0 regardless). Default 1.0 "
             "preserves the legacy behavior for every existing launcher; "
             "lower values (e.g. 0.7) reduce mode-collapse / dead-K on "
             "saturated policies where K=8 rollouts at T=1.0 often produce "
             "all-same outcomes (so groups have zero advantage variance and "
             "contribute zero gradient).",
    )
    parser.add_argument(
        "--sft-adapter-overrides-derived",
        action="store_true",
        help="OPT-IN. When combined with --carry-policy-across-rounds AND "
             "--start-round > 0, the FIRST iterated round loads --sft-adapter "
             "instead of the prefix-derived <prefix>_round{start_round-1:02d}_adapter "
             "path. Needed for cross-run lineage. Default OFF preserves the legacy resume "
             "convention where --sft-adapter is ignored at start_round and the "
             "orchestrator picks up the prior run's saved adapter.",
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
        start_round=args.start_round,
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
        rollout_temperature=float(args.rollout_temperature),
        sft_adapter_overrides_derived=bool(args.sft_adapter_overrides_derived),
    )


# Pre-flight checks


def _preflight(cfg: OrchestrationConfig) -> None:
    """Catch obvious mistakes before spending money on Modal.

    Checks: modal CLI on PATH; config exists and is valid JSON; config's
    replay/ckpt paths match --replay-path/--ckpt-path (else a silent
    split-brain); round count > 0.
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
            "would read another - a silent split-brain. Align them before launching."
        )
    cfg_ckpt = turnrd_cfg.get("ckpt_path")
    if cfg_ckpt and cfg_ckpt != cfg.ckpt_path:
        raise SystemExit(
            f"ERROR: orchestration --ckpt-path={cfg.ckpt_path!r} does not "
            f"match {cfg.config_path}::turnrd.ckpt_path={cfg_ckpt!r}. "
            "Standalone trainer would write a ckpt the parent's refresh "
            "fn never reads. Align them before launching."
        )

    # Architecture version must match between parent and standalone fitter
    # (state_dict keys differ across v1/v2). Reject obviously-wrong values.
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

    # Env-aware eval-task-id range guard. WebShop's goals list is finite
    # (~6910 at num_products=1000); AlfWorld wraps task_id % len(games).
    # Training/eval ranges must be disjoint regardless of env.
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


# Modal command builders


def _to_container_path(local_path: Path) -> str:
    """Translate a local config path to its mounted /workspace/... path.

    image.py mounts the repo root at /workspace inside the container, so host
    absolute paths must be rewritten to be found there.
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
    """Extract the ephemeral app ID (ap-XXXX) from `modal run --detach` output."""
    m = _APP_ID_RE.search(modal_run_stdout)
    return m.group(0) if m else None


def _wait_for_app_finish(
    app_id: str,
    label: str,
    poll_interval_s: float = 10.0,
    timeout_s: float = 3 * 60 * 60,
) -> int:
    """Poll `modal app list` until `app_id` leaves the running state.

    Returns 0 on clean finish, non-zero on timeout/unknown. Default timeout 3h.
    """
    print(f"   polling {app_id} ({label})...")
    t0 = time.time()
    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"   WARNING: timeout waiting for {app_id} after {elapsed:.0f}s.")
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
            print("   WARNING: `modal app list` timed out; retrying.")
            time.sleep(poll_interval_s)
            continue
        if res.returncode != 0:
            print(f"   WARNING: `modal app list` exited {res.returncode}; retrying.")
            time.sleep(poll_interval_s)
            continue
        # Find our row.
        state = None
        for line in res.stdout.splitlines():
            if app_id in line:
                # Strip noise and look for a known lifecycle state.
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
            print(f"   OK {app_id} finished after {elapsed:.0f}s.")
            return 0
        if state is None:
            # Newly-submitted apps may not appear for a few seconds; keep waiting.
            print(
                f"   {app_id} not yet visible in app list (elapsed: {elapsed:.0f}s)..."
            )
        else:
            print(
                f"   {app_id} still {state} (elapsed: {elapsed:.0f}s)..."
            )
        time.sleep(poll_interval_s)


def _has_app_traceback(app_id: str) -> bool:
    """Best-effort detection of an unhandled exception inside the app.

    Modal marks a crashed function as "stopped" too, so lifecycle state alone
    can't tell a crash from a clean exit (this caused a silent R4 cascade once).
    Greps `modal app logs` for known crash signatures. Returns False on any log
    fetch error.
    """
    try:
        res = subprocess.run(
            ["modal", "app", "logs", app_id],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    if res.returncode != 0:
        return False
    blob = (res.stdout or "") + (res.stderr or "")
    # Conservative set of crash signatures we know cascade silently.
    needles = (
        "Traceback (most recent call last)",
        "RuntimeError: CUDA error",
        "HFValidationError",
        "[save_adapter] fallback per-tensor copy also failed",
    )
    return any(n in blob for n in needles)


def _train_loop_cmd(cfg: OrchestrationConfig, round_idx: int) -> list[str]:
    """Build `modal run --detach app_train_loop.py::train_loop_<env> ...`.

    Most knobs come from the JSON --config; only --n-episodes, --k,
    --task-id-offset, and --run-name are overridden per round/seed. --k is read
    from train.K_trajectories_per_task. --config is rewritten to its in-container
    /workspace path. --detach keeps the cloud function alive if the local
    orchestrator dies (we poll via _wait_for_app_finish).
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
    # gpu_mem_util from the JSON train block: caps vLLM KV cache (OOM mitigation).
    gpu_mem_util_cfg = (cfg_json.get("train", {}) or {}).get("gpu_mem_util")
    if gpu_mem_util_cfg is not None:
        cmd.extend(["--gpu-mem-util", str(float(gpu_mem_util_cfg))])
    # sync_every from the JSON train block: aligns vLLM syncs to optimizer steps
    # (default 1 syncs every episode, wasting vLLM's sync budget).
    sync_every_cfg = (cfg_json.get("train", {}) or {}).get("sync_every")
    if sync_every_cfg is not None:
        cmd.extend(["--sync-every", str(int(sync_every_cfg))])
    # Adapter routing:
    # - Legacy: every round loads cfg.sft_adapter (resets to SFT each round).
    # - Carry-policy: R0 loads cfg.sft_adapter; R_N>0 loads R_{N-1}'s saved
    #   adapter and each round writes --save-adapter-out for the next round.
    #   Cross-run opt-in (--sft-adapter-overrides-derived): the first iterated
    #   round loads cfg.sft_adapter instead of the prefix-derived path.
    if cfg.carry_policy_across_rounds:
        cross_run_warm_start = (
            cfg.sft_adapter_overrides_derived
            and round_idx == cfg.start_round
        )
        if round_idx == 0 or cross_run_warm_start:
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
    # Forward rollout_temperature; Modal auto-derives --rollout-temperature
    # from the entrypoint kwarg.
    cmd.extend(["--rollout-temperature", str(float(cfg.rollout_temperature))])
    cmd.extend(cfg.extra_train_loop_args)
    return cmd


def _train_turnrd_cmd(cfg: OrchestrationConfig, round_idx: int = 0) -> list[str]:
    """Build `modal run --detach app_train_turnrd.py ...`.

    Round-independent: reads/writes the same shared Volume paths; replay
    accumulates, ckpt is overwritten. Forwards aux-loss + architecture knobs
    from the JSON so the standalone trainer matches the parent's TurnRD model.
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
    # Architecture: must match build_trainer_from_config or the refresh-fn ckpt
    # load breaks (state_dict key mismatch).
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
        # Recency-decay knobs. Forwarded only when present so configs without
        # them keep legacy behavior (decay disabled).
        ("--recency-decay-half-life", "recency_decay_half_life", None),
        ("--legacy-decay-weight", "legacy_decay_weight", None),
        ("--min-batch-weight", "min_batch_weight", None),
        # LR schedule + fresh-emphasis knobs.
        # Forwarded only when present; legacy default is constant LR.
        ("--warmup-steps", "warmup_steps", None),
        ("--lr-schedule", "lr_schedule", None),
        ("--fresh-emphasis-window-rounds", "fresh_emphasis_window_rounds", None),
        ("--fresh-emphasis-n-epochs", "fresh_emphasis_n_epochs", None),
    ]:
        if jkey in turnrd_block:
            cmd.extend([cli, str(turnrd_block[jkey])])
    # Boolean flags use --flag / --no-flag (not --flag value).
    def _bool_flag(name: str, value: bool) -> list[str]:
        return [f"--{name}" if value else f"--no-{name}"]

    if "causal" in turnrd_block:
        cmd.extend(_bool_flag("causal", bool(turnrd_block["causal"])))
    if "value_head" in turnrd_block:
        cmd.extend(_bool_flag("value-head", bool(turnrd_block["value_head"])))
    if "goal_conditioned_value_head" in turnrd_block:
        cmd.extend(_bool_flag(
            "goal-conditioned-value-head",
            bool(turnrd_block["goal_conditioned_value_head"]),
        ))
    # Cumulative warm-start: when turnrd.cumulative_train is true and round>=1,
    # pass --ckpt-in so the trainer warm-starts from the prior ckpt instead of
    # cold-restarting. Default is legacy cold-start.
    if bool(turnrd_block.get("cumulative_train", False)) and int(round_idx) >= 1:
        cmd.extend(["--ckpt-in", cfg.ckpt_path])
    cmd.extend(cfg.extra_turnrd_args)
    return cmd


# Runner


def _run(cmd: list[str], *, dry_run: bool, label: str) -> int:
    """Submit a detached `modal run`, parse its app ID, then poll until done.

    Returns 0 on success, non-zero on failure. The cloud function keeps running
    even if this orchestrator dies; we trust the cloud state over the local CLI
    exit code (unreliable on transient glitches) whenever an app ID is
    parseable. Only a missing app ID is a hard submission failure.
    """
    print(f"\n-- {label}")
    print(f"   $ {' '.join(cmd)}")
    print("--")
    if dry_run:
        return 0
    t0 = time.time()
    # Phase 1: submit detached. Capture stdout so we can parse the app ID.
    #
    # Blocking subprocess.run (not Popen+SIGTERM): Modal's --detach does not
    # actually decouple the cloud function from the local CLI, so killing the
    # CLI cancels the cloud function. Trade-off: if the heartbeat drops, this
    # hangs; recover by manual kill + resume from the latest saved adapter.
    submit = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True
    )
    print(submit.stdout)
    if submit.stderr:
        print(submit.stderr, file=sys.stderr)
    # Try to parse the app ID from EITHER stream regardless of exit code.
    app_id = _parse_app_id(submit.stdout) or _parse_app_id(submit.stderr)
    if app_id is None:
        # No app ID at all -> the submit truly failed before reaching Modal.
        print(
            f"WARNING: Could not parse app ID from `modal run --detach` output. "
            f"The submit likely failed before the cloud function started "
            f"(CLI exit={submit.returncode}). Aborting this round."
        )
        return submit.returncode if submit.returncode != 0 else 1
    if submit.returncode != 0:
        # CLI exited non-zero but we DO have an app ID. Trust the cloud
        # state instead - Modal CLI's exit code is unreliable on
        # transient network glitches.
        print(
            f"WARNING: Modal CLI exited {submit.returncode} but app {app_id} was "
            f"submitted. Trusting cloud state via polling."
        )
    # Phase 2: poll for completion.
    rc = _wait_for_app_finish(app_id, label=label)
    elapsed = round(time.time() - t0, 2)
    # Phase 3: _wait_for_app_finish only checks lifecycle state, so a crashed
    # function looks like a clean rc=0. Promote to non-zero if logs show a
    # traceback, to stop a silent cascade into the next round.
    if rc == 0 and _has_app_traceback(app_id):
        print(
            f"WARNING: {app_id} state=stopped but logs contain a Python traceback "
            "or known crash signature. Treating this round as FAILED to "
            "prevent silent cascade. Inspect with:\n"
            f"   modal app logs {app_id}"
        )
        rc = 2  # distinct from timeout (124) and submission failure
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

    for round_idx in range(cfg.start_round, cfg.rounds):
        # (a) Parent H-GRPO loop: trains policy + emits replay rows.
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

        # (b) Standalone TurnRD fit on the accumulated replay buffer.
        if round_idx == 0 and cfg.skip_warmup_fit:
            print(
                f"Round {round_idx}: skipping standalone TurnRD fit "
                "(--skip-warmup-fit). Next round's refresh_fn will read "
                f"whatever ckpt is at {cfg.ckpt_path}."
            )
            continue
        rc = _run(
            _train_turnrd_cmd(cfg, round_idx),
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
        f"\n=== Done. {cfg.rounds} rounds x {cfg.episodes_per_round} episodes = "
        f"{cfg.rounds * cfg.episodes_per_round} total H-GRPO episodes. ==="
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])
    _preflight(cfg)
    return _orchestrate(cfg)


if __name__ == "__main__":
    sys.exit(main())
