"""Modal app: generate WebShop SFT trajectories with a deterministic oracle.

  modal run infra/app_webshop_sft_gen.py                               # generate
  modal run infra/app_webshop_sft_gen.py --action summarize             # peek
  modal run infra/app_webshop_sft_gen.py --action inspect               # alias

WebShop has no upstream handcoded expert (unlike AlfWorld's PDDL plan),
so we build a deterministic oracle from the goal-payload primitives
already exposed by `src/envs/webshop_adapter.py` for the
attribute-progress intermediate reward signal:

  * `_extract_target_attrs(env)`  -> set[str] target attr tokens.
  * `_extract_target_asin(env)`   -> str|None target product ASIN.
  * `_extract_selected_attrs(env)` -> currently engaged options.

Per-episode oracle policy:
  1. Reset -> snapshot `env.goal["query"]`, `env.goal["asin"]` (or
     `goal["asins"][0]`), `env.goal["attributes"]`.
  2. Emit `search[<query>]`.
  3. Up to N=5 search-result pages: if `target_asin` substring appears
     in the rendered obs, emit `click[<asin>]` and move on; else emit
     `click[Next >]` and recurse. If never found, drop the trajectory.
  4. On the item page, for each canonicalised goal attribute token, try
     `click[<token>]` (best-effort - values not surfaced as options
     are simply no-ops). Cap at len(attributes)+a few safety turns.
  5. Emit `click[Buy Now]`.

Terminal env reward is the attribute-match fraction; only trajectories
whose final reward >= `reward_threshold` (default 0.99) get flushed.
Partial demos are dropped wholesale rather than written to disk.

Output schema (one row per turn, single JSONL file):
    {"prompt": str, "action": str, "instruction": str, "step_idx": int,
     "trajectory_id": str, "final_reward": float}

`prompt` is pre-rendered via the runtime renderer
`src.envs.prompts.react_webshop.render_webshop_turn_prompt` - the SAME
module the runtime ReAct collector uses + the SAME module that the
SFT loader's `default_render_prompt` shim delegates to. That guarantees
SFT-rollout prompt-parity (the v3 R=0 template-drift root cause).

`--include-human-trajs` additionally concatenates the upstream WebShop
~50 gdown human trajectories (loaded via
`load_sft_examples_from_directory`) into the same JSONL, so the
trainer has one input file.

Cost: ~$0.50, ~30 min CPU at n_sessions=2000.
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import webshop_image

app = modal.App("cs224r-hgpo-webshop-sft-gen")


# Where the oracle gen app writes the SFT trajectories on the shared
# Volume. Single JSONL (AlfWorld pattern) - vs the gdown human-trajs
# directory-of-files layout. The trainer's `--data-path` flag dispatches
# on suffix (.jsonl -> single-file loader) so this path Just Works.
SFT_OUTPUT_PATH = "/vol/data/webshop/oracle_trajs.jsonl"

# Where the upstream WebShop gdown human-trajectory dump lives on the
# Volume (one .jsonl per trajectory). Used by `--include-human-trajs`
# to pre-render those into the same merged JSONL.
HUMAN_TRAJS_DIR = "/vol/data/webshop/human_trajs"


def _resolve_env_goal(env) -> dict | None:
    """Pull the per-session goal dict from the upstream WebShop env.

    Mirrors the tiered access pattern used by the introspection helpers
    in `src/envs/webshop_adapter.py`. Returns None if no goal is
    accessible (the caller drops the trajectory).
    """
    goal = getattr(env, "goal", None)
    if isinstance(goal, dict):
        return goal
    server = getattr(env, "server", None)
    goals_dict = getattr(server, "goals", None) if server is not None else None
    session = getattr(env, "session", None)
    if isinstance(goals_dict, dict) and session is not None:
        return goals_dict.get(session)
    if isinstance(goals_dict, list):
        try:
            return goals_dict[int(session)] if session is not None else None
        except (TypeError, ValueError, IndexError):
            return None
    return None


def _instruction_from_goal(goal: dict | None) -> str:
    if not isinstance(goal, dict):
        return ""
    for key in ("instruction_text", "query"):
        val = goal.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _query_from_goal(goal: dict | None) -> str:
    """Build the BM25 search query the oracle should issue.

    WebShop caps `search[...]` results at 50 (5 pages x 10/page) so the
    query must be SELECTIVE enough to surface the target ASIN inside
    the top-50 BM25 hits. The goal payload exposes (in increasing
    order of BM25 selectivity):

      * `goal["query"]` - short user-typed category, e.g.
        "men's polos". 0/N oracle win rate on real env (way too
        generic - top-50 BM25 hits don't include the target ASIN).
      * `goal["instruction_text"]` - verbose generated description
        listing every required attribute, e.g. "Find me slim fit,
        moisture wicking men's polos with short sleeve, ...".
        Marginally better but attribute tokens (sizes, materials) can
        BM25-dominate and pull the wrong category of products
        (e.g. "underwater photography ... 8x6.5ft" pulls 8x6.5ft
        BACKDROPS instead of actual photography equipment).
      * `goal["name"]` - the target product's actual title. Pinpoint
        precision: searching for the title returns the exact target
        ASIN as the top BM25 hit virtually 100% of the time.

    Prefers `name`, falls back to `instruction_text`, then `query`,
    then empty. Empirically this is the difference between ~0% and
    ~95% oracle win rate against the 1k-product split.
    """
    if not isinstance(goal, dict):
        return ""
    name = goal.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    instr = goal.get("instruction_text")
    if isinstance(instr, str) and instr.strip():
        return instr.strip()
    q = goal.get("query")
    if isinstance(q, str) and q.strip():
        return q.strip()
    return ""


def _render_prompt_for_state(state, history: list, instruction: str) -> str:
    """Render a WebShop ReAct prompt for `state` + `history` + `instruction`.

    Wraps `state` in a SimpleNamespace shim that exposes `instruction`
    (the runtime WebShopState dataclass doesn't carry it directly) so
    the renderer's `state.instruction` access path resolves. The shim
    is byte-identical in semantics to what
    `src/datasets/sft_webshop.py::default_render_prompt` does on the
    SFT loader side.
    """
    from types import SimpleNamespace

    from src.envs.prompts.react_webshop import render_webshop_turn_prompt

    shim = SimpleNamespace(
        observation_text=getattr(state, "observation_text", "") or "",
        instruction=instruction,
        valid_actions=list(getattr(state, "valid_actions", []) or []),
    )
    return render_webshop_turn_prompt(shim, history)


def _oracle_episode(
    adapter,
    session_id: int,
    *,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.99,
    infos_out: list | None = None,
) -> tuple[list[dict], float, str]:
    """Run one oracle episode; return (traj_rows, final_reward, status).

    `traj_rows` is empty when the trajectory is dropped (oracle failed
    to win the episode). `status` is one of:
      - "won"             : terminal reward >= threshold; rows kept.
      - "no_goal"         : env didn't expose a recognisable goal.
      - "no_target_asin"  : goal had no ASIN field.
      - "asin_not_found"  : target ASIN never appeared in the first
                            `max_result_pages` of search results.
      - "lost"            : env terminated with reward < threshold.
      - "truncated"       : hit `max_steps_per_episode` before terminal.
      - "exception:<E>"   : an env step raised an exception.

    If `infos_out` is supplied, each step's post-step `info` dict is
    appended (including `intermediate_reward` /
    `intermediate_reward_source` when the dense-signal flag is on at
    adapter construction). The list is populated regardless of final
    status (so callers can inspect partial/dropped trajectories too).
    """
    from types import SimpleNamespace

    try:
        state = adapter.reset(task_id=session_id)
    except Exception as exc:
        return [], 0.0, f"exception:reset:{type(exc).__name__}"

    goal = _resolve_env_goal(adapter._env)
    if goal is None:
        return [], 0.0, "no_goal"

    instruction = _instruction_from_goal(goal)
    query = _query_from_goal(goal)
    if not query:
        return [], 0.0, "no_goal"

    target_asin: str | None = None
    g_asin = goal.get("asin")
    if isinstance(g_asin, str) and g_asin.strip():
        target_asin = g_asin.strip().lower()
    else:
        asins = goal.get("asins")
        if isinstance(asins, (list, tuple)) and asins and isinstance(asins[0], str):
            target_asin = asins[0].strip().lower()
    if not target_asin:
        return [], 0.0, "no_target_asin"

    attrs_raw = goal.get("attributes") or []
    if not isinstance(attrs_raw, (list, tuple)):
        attrs_raw = []
    # Build the click-targets list. WebShop item pages render option
    # buttons whose labels are the EXACT option values, not the goal's
    # short attribute tokens. Two goal fields contribute:
    #
    #   * `goal["goal_options"]` - dict like {"color": "dark blue",
    #     "size": "x-large"} mapping option NAME -> exact VALUE the
    #     env's reward function expects. These values match the
    #     item-page button labels byte-for-byte -> click[<value>] is
    #     the precise option-engagement signal.
    #   * `goal["attributes"]` - flat list of attribute tokens like
    #     "moisture wicking", "stretch fabric". Some of these surface
    #     as item-page buttons (then click[<token>] works), some don't
    #     (they're hidden product-description attributes the env
    #     credits when the target ASIN is bought). Tried as a
    #     supplement to goal_options for best-effort coverage.
    #
    # Preserve original casing (button labels render mixed case) and
    # de-duplicate while preserving order.
    seen: set[str] = set()
    target_attrs: list[str] = []
    goal_options = goal.get("goal_options")
    if isinstance(goal_options, dict):
        for v in goal_options.values():
            if not isinstance(v, str):
                continue
            canon = v.strip()
            if not canon:
                continue
            key = canon.lower()
            if key in seen:
                continue
            seen.add(key)
            target_attrs.append(canon)
    for a in attrs_raw:
        if not isinstance(a, str):
            continue
        canon = a.strip()
        if not canon:
            continue
        key = canon.lower()
        if key in seen:
            continue
        seen.add(key)
        target_attrs.append(canon)

    history: list = []
    traj_rows: list[dict] = []
    traj_id = f"oracle_{session_id:06d}"

    def _record_and_step(action: str) -> tuple:
        """Render + record one (prompt, action) pair, then env.step it.

        Returns the post-step (state, reward, done, info) tuple plus a
        bool indicating whether an exception fired.
        """
        nonlocal state, history
        prompt = _render_prompt_for_state(state, history, instruction)
        traj_rows.append({
            "prompt": prompt,
            "action": action,
            "step_idx": len(traj_rows),
            "trajectory_id": traj_id,
            "instruction": instruction,
            # final_reward backfilled on flush.
            "final_reward": 0.0,
        })
        history.append(SimpleNamespace(
            observation_text=getattr(state, "observation_text", "") or "",
            action_text=action,
        ))
        try:
            new_state, reward, done, info = adapter.step(action)
        except Exception as exc:
            return None, 0.0, True, {"_exc": f"{type(exc).__name__}:{exc}"}, True
        if infos_out is not None:
            # Snapshot the post-step info dict so the caller can inspect
            # `intermediate_reward` / `intermediate_reward_source` per turn
            # without having to re-implement the oracle loop.
            infos_out.append({
                "action": action,
                "raw_env_reward": float(reward),
                "done": bool(done),
                "info": dict(info) if isinstance(info, dict) else {},
            })
        state = new_state
        return new_state, float(reward), bool(done), info, False

    # step 1: initial search
    action = f"search[{query}]"
    out = _record_and_step(action)
    if out[-1]:
        return [], 0.0, f"exception:step:{out[3].get('_exc', '?')}"
    _, reward, done, info, _ = out
    if done:
        # Premature done on a search? Treat as lost unless wonky.
        return ([] if reward < reward_threshold else traj_rows), reward, (
            "won" if reward >= reward_threshold else "lost"
        )

    # step 2-N: walk search result pages, find target ASIN
    asin_clicked = False
    for _ in range(max_result_pages):
        if len(traj_rows) >= max_steps_per_episode:
            return [], 0.0, "truncated"
        obs_text = (getattr(state, "observation_text", "") or "").lower()
        if target_asin in obs_text:
            action = f"click[{target_asin}]"
            out = _record_and_step(action)
            if out[-1]:
                return [], 0.0, f"exception:step:{out[3].get('_exc', '?')}"
            _, reward, done, info, _ = out
            asin_clicked = True
            if done:
                return ([] if reward < reward_threshold else traj_rows), reward, (
                    "won" if reward >= reward_threshold else "lost"
                )
            break
        # Not on this page -> next page.
        action = "click[Next >]"
        out = _record_and_step(action)
        if out[-1]:
            return [], 0.0, f"exception:step:{out[3].get('_exc', '?')}"
        _, reward, done, info, _ = out
        if done:
            return ([] if reward < reward_threshold else traj_rows), reward, (
                "won" if reward >= reward_threshold else "lost"
            )

    if not asin_clicked:
        return [], 0.0, "asin_not_found"

    # step N+1..: best-effort attribute option clicks
    for attr in target_attrs:
        if len(traj_rows) >= max_steps_per_episode:
            return [], 0.0, "truncated"
        action = f"click[{attr}]"
        out = _record_and_step(action)
        if out[-1]:
            # Some option strings can crash the upstream `text_to_acts`
            # parser; treat as a soft no-op by dropping this attempt
            # but continuing.
            continue
        _, reward, done, info, _ = out
        if done:
            return ([] if reward < reward_threshold else traj_rows), reward, (
                "won" if reward >= reward_threshold else "lost"
            )

    # terminal: click Buy Now
    if len(traj_rows) >= max_steps_per_episode:
        return [], 0.0, "truncated"
    out = _record_and_step("click[Buy Now]")
    if out[-1]:
        return [], 0.0, f"exception:step:{out[3].get('_exc', '?')}"
    _, reward, done, info, _ = out

    if reward >= reward_threshold:
        return traj_rows, reward, "won"
    return [], reward, "lost"


def _add_local_render_paths():
    """Append the WebShop pyuser site-packages + repo dir to sys.path.

    Needed inside the gen container so `import web_agent_site...`
    resolves against the editable install dropped by
    `app_webshop_install.py::pip_install_webshop`.
    """
    import os
    import sys

    pyuser = "/vol/webshop_pyuser"
    repo = "/vol/code/webshop"
    if repo not in sys.path:
        sys.path.insert(0, repo)
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = os.path.join(pyuser, "lib", pyver, "site-packages")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    if "/workspace" not in sys.path:
        sys.path.insert(0, "/workspace")


@app.function(
    image=webshop_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=2 * 60 * 60,  # 2 h cap; expected runtime ~30-60 min CPU
)
def generate_sft_trajectories(
    n_sessions: int = 2000,
    output_path: str = SFT_OUTPUT_PATH,
    *,
    session_id_base: int = 0,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.99,
    include_human_trajs: bool = False,
    human_trajs_dir: str = HUMAN_TRAJS_DIR,
    human_trajs_min_reward: float = 0.5,
) -> dict:
    """Iterate session ids [base, base+n_sessions), drive oracle, write JSONL.

    Args:
        n_sessions: how many distinct sessions to attempt. WebShop's
                    upstream env picks the goal via `reset(session=i)`.
        output_path: where to write the JSONL on the shared Volume.
                     Truncated on entry - re-running this app produces
                     a fresh dataset.
        session_id_base: starting session id. Default 0 mirrors the
                         upstream's training-split start. Bumps useful
                         if you want to dedupe against the eval slice
                         `[6500, 6600)`.
        max_result_pages: how many search-result pages to scan for the
                          target ASIN before giving up on the session.
        max_steps_per_episode: hard cap on adapter steps per episode.
        reward_threshold: minimum terminal env reward to keep a
                          trajectory. 0.99 ~ "all target attrs matched".
        include_human_trajs: when True, also pre-render the
                             ~50 upstream gdown human trajectories at
                             `human_trajs_dir` and append them to the
                             same JSONL. Uses
                             `load_sft_examples_from_directory` so the
                             renderer is the SAME runtime ReAct one.
        human_trajs_dir: dir of `<traj>.jsonl` files. Defaults to the
                         path written by `app_data.py::download_human_trajectories`.
        human_trajs_min_reward: drop human trajs with final reward
                                below this. Default 0.5 matches the
                                current trainer's `--min-reward` knob.

    Returns:
        manifest dict with cardinalities + per-status breakdown.
    """
    import json
    import os
    import time
    from collections import Counter
    from pathlib import Path

    _add_local_render_paths()

    from src.envs.webshop_adapter import WebShopAdapter

    print(f">>> Building WebShopAdapter for n_sessions={n_sessions} "
          f"starting at session={session_id_base}")
    adapter = WebShopAdapter(
        max_steps=max_steps_per_episode,
        observation_mode="text",
        task_split="train",
        env_kwargs={},
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(output_path, "w")

    status_counts: Counter[str] = Counter()
    won_rewards: list[float] = []
    lost_rewards: list[float] = []
    n_won = 0
    n_examples = 0
    t0 = time.time()

    for i in range(n_sessions):
        sid = session_id_base + i
        try:
            traj_rows, final_reward, status = _oracle_episode(
                adapter,
                sid,
                max_result_pages=max_result_pages,
                max_steps_per_episode=max_steps_per_episode,
                reward_threshold=reward_threshold,
            )
        except Exception as exc:  # pragma: no cover - defensive
            status = f"exception:outer:{type(exc).__name__}"
            traj_rows = []
            final_reward = 0.0

        status_counts[status] += 1
        if status == "won" and traj_rows:
            n_won += 1
            won_rewards.append(final_reward)
            for row in traj_rows:
                row["final_reward"] = float(final_reward)
                fh.write(json.dumps(row) + "\n")
                n_examples += 1
            fh.flush()
        elif status == "lost":
            lost_rewards.append(final_reward)

        if (i + 1) % 25 == 0:
            elapsed = round(time.time() - t0, 1)
            win_rate = n_won / (i + 1)
            print(
                f"  [{i + 1}/{n_sessions}] won={n_won} ({win_rate:.1%}) "
                f"examples={n_examples} t={elapsed}s "
                f"status_counts={dict(status_counts)}"
            )

    n_oracle_examples = n_examples

    # optional: pre-render & concat upstream human trajs
    n_human_examples = 0
    if include_human_trajs:
        if not os.path.isdir(human_trajs_dir):
            print(
                f"  WARN: include_human_trajs=True but {human_trajs_dir!r} "
                "does not exist; skipping. Run "
                "infra/app_data.py --action download_human_trajectories first."
            )
        else:
            from src.datasets.sft_webshop import (
                load_sft_examples_from_directory,
            )

            print(f">>> Concatenating human trajs from {human_trajs_dir} "
                  f"(min_reward={human_trajs_min_reward})")
            human_examples = load_sft_examples_from_directory(
                human_trajs_dir, min_reward=human_trajs_min_reward
            )
            for ex in human_examples:
                fh.write(json.dumps({
                    "prompt": ex.prompt,
                    "action": ex.action,
                    "step_idx": ex.step_idx,
                    "trajectory_id": f"human_{ex.trajectory_id}",
                    "instruction": ex.instruction,
                    "final_reward": float(ex.final_reward),
                }) + "\n")
                n_human_examples += 1
            print(f">>> Human-traj examples appended: {n_human_examples}")
            n_examples += n_human_examples

    fh.close()
    volume.commit()

    manifest = {
        "output_path": output_path,
        "n_sessions_attempted": n_sessions,
        "session_id_base": session_id_base,
        "n_oracle_won": n_won,
        "oracle_win_rate": round(n_won / max(1, n_sessions), 4),
        "n_oracle_examples": n_oracle_examples,
        "n_human_examples": n_human_examples,
        "n_examples_written": n_examples,
        "mean_won_final_reward": (
            round(sum(won_rewards) / max(1, len(won_rewards)), 4)
        ),
        "won_reward_min": min(won_rewards) if won_rewards else 0.0,
        "won_reward_max": max(won_rewards) if won_rewards else 0.0,
        "mean_lost_final_reward": (
            round(sum(lost_rewards) / max(1, len(lost_rewards)), 4)
        ),
        "lost_reward_min": min(lost_rewards) if lost_rewards else 0.0,
        "lost_reward_max": max(lost_rewards) if lost_rewards else 0.0,
        "status_counts": dict(status_counts),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(">>> SFT-gen done:", manifest)
    return manifest


@app.function(
    image=webshop_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=10 * 60,
)
def summarize_sft_dataset_app(path: str = SFT_OUTPUT_PATH) -> dict:
    """Peek at the SFT JSONL: cardinality + first row preview.

    Useful smoke test after `generate_sft_trajectories` to confirm the
    file is non-trivial before kicking off the SFT trainer.
    """
    import json
    import os

    _add_local_render_paths()

    if not os.path.isfile(path):
        return {"path": path, "exists": False}

    n = 0
    n_trajectories: set[str] = set()
    rewards: list[float] = []
    by_action_kind: dict[str, int] = {}
    first_row: dict | None = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if first_row is None:
                first_row = row
            n += 1
            traj_id = row.get("trajectory_id")
            if isinstance(traj_id, str):
                n_trajectories.add(traj_id)
            act = str(row.get("action", ""))
            head = act.split("[", 1)[0]
            by_action_kind[head] = by_action_kind.get(head, 0) + 1
            try:
                rewards.append(float(row.get("final_reward", 0.0)))
            except (TypeError, ValueError):
                pass

    summary = {
        "n_examples": n,
        "n_trajectories": len(n_trajectories),
        "by_action_kind": by_action_kind,
        "mean_final_reward": round(sum(rewards) / max(1, len(rewards)), 4),
        "min_final_reward": min(rewards) if rewards else 0.0,
        "max_final_reward": max(rewards) if rewards else 0.0,
    }
    preview = None
    if first_row is not None:
        prompt = str(first_row.get("prompt", ""))
        preview = {
            "prompt_head": prompt[:600],
            "prompt_tail": prompt[-200:],
            "action": first_row.get("action"),
            "trajectory_id": first_row.get("trajectory_id"),
            "step_idx": first_row.get("step_idx"),
            "instruction_head": str(first_row.get("instruction", ""))[:200],
            "final_reward": first_row.get("final_reward"),
        }
    return {
        "path": path,
        "exists": True,
        "summary": summary,
        "first_row": preview,
    }


@app.function(
    image=webshop_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=10 * 60,
)
def diagnose_oracle(
    n_sessions: int = 3,
    session_id_base: int = 0,
    max_result_pages: int = 10,
) -> dict:
    """Diagnostic: for each session, dump goal payload + each search-result
    page's obs so we can see WHY the oracle can't find the target ASIN.

    For each session:
      - goal.query / goal.asin / goal.attributes / goal.category
      - obs of `search[goal.query]`
      - obs of up to `max_result_pages-1` subsequent `click[Next >]`
      - on each page: bool whether goal.asin substring is in the obs

    Surfaces the typical failure mode: BM25 search returns top-K results
    that don't include the target ASIN within `max_result_pages` pages.
    Helps decide if we need to (a) increase max_result_pages, (b) make
    the oracle's search query more selective, or (c) drop the
    attribute-engagement step.
    """
    _add_local_render_paths()

    from src.envs.webshop_adapter import WebShopAdapter

    adapter = WebShopAdapter(
        max_steps=max_result_pages + 2,
        observation_mode="text",
        task_split="train",
        env_kwargs={},
    )

    report: list[dict] = []
    for i in range(n_sessions):
        sid = session_id_base + i
        try:
            state = adapter.reset(task_id=sid)
        except Exception as exc:
            report.append({"sid": sid, "error": f"reset:{type(exc).__name__}:{exc}"})
            continue

        goal = _resolve_env_goal(adapter._env) or {}
        target_asin = None
        g_asin = goal.get("asin")
        if isinstance(g_asin, str) and g_asin.strip():
            target_asin = g_asin.strip().lower()
        else:
            asins = goal.get("asins")
            if isinstance(asins, (list, tuple)) and asins and isinstance(asins[0], str):
                target_asin = asins[0].strip().lower()

        query = _query_from_goal(goal)
        per_page: list[dict] = []
        # Page 1: issue search.
        try:
            state, reward, done, info = adapter.step(f"search[{query}]")
        except Exception as exc:
            per_page.append({"page": 1, "error": f"step:{type(exc).__name__}:{exc}"})
            report.append({"sid": sid, "goal": _goal_summary(goal), "pages": per_page})
            continue

        obs_p1 = (state.observation_text or "")
        per_page.append({
            "page": 1,
            "action": f"search[{query}]",
            "obs_head": obs_p1[:600],
            "obs_len": len(obs_p1),
            "asin_in_obs": (target_asin in obs_p1.lower()) if target_asin else None,
        })

        for page_idx in range(2, max_result_pages + 1):
            if done:
                break
            try:
                state, reward, done, info = adapter.step("click[Next >]")
            except Exception as exc:
                per_page.append({"page": page_idx, "error": f"step:{type(exc).__name__}:{exc}"})
                break
            obs_pi = (state.observation_text or "")
            per_page.append({
                "page": page_idx,
                "action": "click[Next >]",
                "obs_head": obs_pi[:600],
                "obs_len": len(obs_pi),
                "asin_in_obs": (target_asin in obs_pi.lower()) if target_asin else None,
            })

        report.append({
            "sid": sid,
            "goal": _goal_summary(goal),
            "target_asin": target_asin,
            "n_pages_scanned": len(per_page),
            "asin_seen_on_page": next(
                (p["page"] for p in per_page if p.get("asin_in_obs")), None,
            ),
            "pages": per_page,
        })

    return {"n_sessions": n_sessions, "report": report}


def _goal_summary(goal: dict | None) -> dict:
    """Compact representation of the goal payload for diagnostic logs."""
    if not isinstance(goal, dict):
        return {}
    return {
        "instruction_text": (goal.get("instruction_text") or "")[:200],
        "query": goal.get("query"),
        "asin": goal.get("asin"),
        "asins": goal.get("asins"),
        "attributes": goal.get("attributes"),
        "category": goal.get("category"),
        "price_upper": goal.get("price_upper"),
        "all_keys": sorted(list(goal.keys())),
    }


def _classify_webshop_action(act: str, target_asin: str | None) -> str:
    """Bucket a WebShop ReAct action for IR-distribution analysis."""
    if act.startswith("search["):
        return "search"
    if target_asin and act.lower() == f"click[{target_asin}]":
        return "click_target_asin"
    if act == "click[Buy Now]":
        return "click_buy_now"
    if act in {"click[Next >]", "click[< Prev]", "click[Back to Search]"}:
        return "click_nav"
    if act.startswith("click["):
        return "click_option"
    return "other"


@app.function(
    image=webshop_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=30 * 60,
)
def validate_dense_signal(
    n_sessions: int = 10,
    session_id_base: int = 0,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.5,
    sample_traj_idx: int = 0,
) -> dict:
    """Validate WebShop's per-turn `intermediate_reward` dense signal.

    WebShop has NO PDDL facts-diff signal (unlike AlfWorld) - its only
    per-turn supervision for TurnRDv2's V-head is the attribute-progress
    intermediate reward synthesised by `WebShopAdapter.step()` when
    `use_attribute_progress_intermediate_reward=True` (the flag set in
    all 3 RL configs' env block). This validator drives the oracle
    against the live env WITH that flag on, captures each step's
    `info["intermediate_reward"]`, and reports:

      * `n_sessions_with_signal`: how many sessions had at least one
        non-None IR (indicator that target-attrs introspection
        worked at reset).
      * `signal_coverage`: across captured turns, fraction with
        non-None IR. ~1.0 means the signal never silently degraded;
        < 1.0 means the upstream env occasionally hid the goal-attr
        payload mid-trajectory.
      * `mean_ir_by_action_kind`: per-bucket means. Expected pattern
        (this is what makes the signal HIGH QUALITY for TurnRD
        supervision):
          - search ~ 0.0           (no option engagement)
          - click_nav ~ 0.0        (Next/Prev/Back has no IR delta)
          - click_target_asin ~ 0.25 (one-time ASIN-landing bonus)
          - click_option ~ 1/|target_attrs| (per-attribute delta when
            the click hits an option in the goal-options set)
          - click_buy_now ~ 0.0    (terminal; no further engagement)
      * `cum_ir_per_session`: per-session (cumulative_IR, final_raw_reward)
        pairs. CORRELATION between cum_IR and final_raw_reward is the
        single most diagnostic metric - if it's > 0.7, the signal is
        carrying real per-turn credit; if it's ~ 0, the signal is
        noise and TurnRD's V-head will learn nothing.
      * `pearson_r`: Pearson correlation between cum_IR and final
        reward across captured sessions.
      * `sample_trajectory`: full per-turn (action, raw_env_reward,
        intermediate_reward, intermediate_reward_source, target_overlap_after)
        dump for visual sanity inspection.

    This exercises the EXACT path the producer (collector) reads from
    at `src/algorithms/grpo/collectors.py:251`
    (`info.get("intermediate_reward")`), so if the signal looks right
    here, TurnRDv2 will see the same shape in the replay JSONL.
    """
    _add_local_render_paths()

    from src.envs.webshop_adapter import WebShopAdapter

    # CRITICAL: build the adapter with the dense-signal flag ENABLED.
    # This is the only difference from the SFT-gen path - same env,
    # same goal payloads, same oracle action sequence; just an extra
    # `info["intermediate_reward"]` field per step.
    adapter = WebShopAdapter(
        max_steps=max_steps_per_episode,
        observation_mode="text",
        task_split="train",
        env_kwargs={},
        use_attribute_progress_intermediate_reward=True,
    )

    per_session_records: list[dict] = []
    bucket_sums: dict[str, float] = {}
    bucket_counts: dict[str, int] = {}
    n_signal_turns = 0
    n_total_turns = 0
    n_sessions_with_signal = 0
    sample_trajectory: list[dict] | None = None

    for i in range(n_sessions):
        sid = session_id_base + i

        # Re-resolve target_asin BEFORE the oracle clobbers state, so
        # we can bucket actions (e.g. tell click_target_asin from
        # click_option) without re-introspecting the env later.
        try:
            adapter.reset(task_id=sid)
        except Exception:
            continue
        goal = _resolve_env_goal(adapter._env) or {}
        target_asin = None
        g_asin = goal.get("asin")
        if isinstance(g_asin, str) and g_asin.strip():
            target_asin = g_asin.strip().lower()
        elif isinstance(goal.get("asins"), (list, tuple)) and goal["asins"]:
            first = goal["asins"][0]
            if isinstance(first, str):
                target_asin = first.strip().lower()
        n_target_attrs = (
            len(adapter._target_attrs) if adapter._target_attrs else 0
        )

        # Now run the actual oracle episode with info capture. The
        # adapter does its own reset(task_id=sid) at the top, so the
        # pre-fetch above doesn't leak state.
        infos: list = []
        traj_rows, final_reward, status = _oracle_episode(
            adapter,
            sid,
            max_result_pages=max_result_pages,
            max_steps_per_episode=max_steps_per_episode,
            reward_threshold=reward_threshold,
            infos_out=infos,
        )

        # Per-turn records for this session.
        turn_records: list[dict] = []
        cum_ir = 0.0
        any_signal = False
        for info_rec in infos:
            n_total_turns += 1
            ir = info_rec["info"].get("intermediate_reward")
            ir_src = info_rec["info"].get("intermediate_reward_source")
            if ir is not None:
                n_signal_turns += 1
                any_signal = True
                cum_ir += float(ir)
            action = info_rec["action"]
            bucket = _classify_webshop_action(action, target_asin)
            bucket_sums[bucket] = bucket_sums.get(bucket, 0.0) + (
                float(ir) if ir is not None else 0.0
            )
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            turn_records.append({
                "turn": len(turn_records),
                "action": action,
                "action_kind": bucket,
                "raw_env_reward": info_rec["raw_env_reward"],
                "done": info_rec["done"],
                "intermediate_reward": (
                    None if ir is None else round(float(ir), 4)
                ),
                "intermediate_reward_source": ir_src,
                "n_target_attrs": n_target_attrs,
            })
        if any_signal:
            n_sessions_with_signal += 1
        per_session_records.append({
            "sid": sid,
            "status": status,
            "final_raw_reward": float(final_reward),
            "cum_intermediate_reward": round(cum_ir, 4),
            "n_turns": len(turn_records),
            "n_target_attrs": n_target_attrs,
        })
        if i == sample_traj_idx and turn_records:
            sample_trajectory = turn_records

    # Aggregate stats.
    mean_ir_by_kind = {
        k: round(bucket_sums[k] / max(1, bucket_counts[k]), 4)
        for k in sorted(bucket_sums.keys())
    }
    bucket_n = {k: bucket_counts[k] for k in sorted(bucket_counts.keys())}

    # Pearson r between cum_IR and final_raw_reward across sessions
    # (sessions with at least one captured turn).
    pairs = [
        (s["cum_intermediate_reward"], s["final_raw_reward"])
        for s in per_session_records if s["n_turns"] > 0
    ]
    pearson_r: float | None = None
    if len(pairs) >= 2:
        n = len(pairs)
        sx = sum(p[0] for p in pairs)
        sy = sum(p[1] for p in pairs)
        sxx = sum(p[0] * p[0] for p in pairs)
        syy = sum(p[1] * p[1] for p in pairs)
        sxy = sum(p[0] * p[1] for p in pairs)
        denom = ((n * sxx - sx * sx) * (n * syy - sy * sy)) ** 0.5
        if denom > 1e-9:
            pearson_r = round((n * sxy - sx * sy) / denom, 4)

    return {
        "n_sessions_attempted": n_sessions,
        "n_sessions_with_signal": n_sessions_with_signal,
        "signal_coverage": (
            round(n_signal_turns / max(1, n_total_turns), 4)
        ),
        "n_total_turns": n_total_turns,
        "n_signal_turns": n_signal_turns,
        "mean_ir_by_action_kind": mean_ir_by_kind,
        "turn_count_by_action_kind": bucket_n,
        "pearson_r_cum_ir_vs_final_reward": pearson_r,
        "per_session_summary": per_session_records,
        "sample_trajectory": sample_trajectory,
    }


@app.local_entrypoint()
def main(
    action: str = "generate",
    n_sessions: int = 2000,
    output_path: str = SFT_OUTPUT_PATH,
    session_id_base: int = 0,
    max_result_pages: int = 5,
    max_steps_per_episode: int = 25,
    reward_threshold: float = 0.99,
    include_human_trajs: bool = False,
    human_trajs_dir: str = HUMAN_TRAJS_DIR,
    human_trajs_min_reward: float = 0.5,
) -> None:
    """Local entrypoint dispatching on `--action`.

    Actions:
      generate            - run `generate_sft_trajectories(...)`
      summarize / inspect - pretty-print the dataset summary + first row
    """
    import json as _json

    if action == "generate":
        res = generate_sft_trajectories.remote(
            n_sessions=n_sessions,
            output_path=output_path,
            session_id_base=session_id_base,
            max_result_pages=max_result_pages,
            max_steps_per_episode=max_steps_per_episode,
            reward_threshold=reward_threshold,
            include_human_trajs=include_human_trajs,
            human_trajs_dir=human_trajs_dir,
            human_trajs_min_reward=human_trajs_min_reward,
        )
    elif action in {"summarize", "inspect"}:
        res = summarize_sft_dataset_app.remote(path=output_path)
    elif action == "diagnose":
        res = diagnose_oracle.remote(
            n_sessions=n_sessions,
            session_id_base=session_id_base,
            max_result_pages=max_result_pages,
        )
    elif action == "validate_signal":
        res = validate_dense_signal.remote(
            n_sessions=n_sessions,
            session_id_base=session_id_base,
            max_result_pages=max_result_pages,
            max_steps_per_episode=max_steps_per_episode,
            reward_threshold=reward_threshold,
        )
    else:
        raise ValueError(
            f"Unknown action {action!r}. "
            "Expected one of: generate, summarize, inspect, "
            "diagnose, validate_signal."
        )
    print(_json.dumps(res, indent=2, default=str))
