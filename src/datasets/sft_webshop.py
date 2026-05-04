"""SFT dataset loader for the WebShop human-trajectory JSONL files.

Each `.jsonl` file is one human shopping trajectory:
    row 0:  page=index, url=<base>/<task_id>                                 (initial state)
    row k:  page=search_results, url=<base>/search_results/<task>/<q>/<pg>   (after a search)
    row j:  page=item_page,      url=<base>/item_page/<task>/<asin>/<q>/...  (after a click)
    row N:  page=done,           url=<base>/done/<task>/<asin>/<opts>        (terminal; reward populated)

Actions are encoded in the URL of the *next* row, so we extract them by
diffing successive URLs. We produce SFTExamples of the form (prompt,
target_action) suitable for cross-entropy SFT against a Qwen-style
chat template.

This module is pure Python (stdlib only) so it's unit-testable locally
without torch / transformers.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import unquote


@dataclass(frozen=True)
class SFTExample:
    """One (prompt, target_action) training pair."""

    prompt: str
    action: str
    instruction: str
    step_idx: int
    trajectory_id: str
    final_reward: float


# --------- URL parsing ----------------------------------------------------


def _path_segments(url: str) -> list[str]:
    """Split URL path into segments after the host:port; strip empties."""
    if "://" in url:
        url = url.split("://", 1)[1]
    if "/" in url:
        url = url.split("/", 1)[1]
    return [s for s in url.split("/") if s]


def _decode_query_list(encoded: str) -> list[str]:
    """The URL-encoded query is a python-list-literal like
    ``%5B%27red%27%2C%20%27dress%27%5D`` → ``['red', 'dress']``. Decode
    + ast.literal_eval.
    """
    try:
        decoded = unquote(encoded)
        val = ast.literal_eval(decoded)
        if isinstance(val, list):
            return [str(x) for x in val]
    except Exception:
        pass
    return [encoded]


def _action_from_url_transition(prev_url: str, next_url: str) -> str | None:
    """Infer the WebShop ReAct action from `prev_url` → `next_url`.

    Diffs the two URLs to disambiguate transitions that look superficially
    identical but represent different agent actions (review item A8):

      * index → search_results              → search[<query>]
      * search_results page=N → page=N+1, same query → click[Next >]
      * search_results page=N → page=N-1, same query → click[< Prev]
      * search_results → search_results, different query → search[<new>]
      * search_results → item_page          → click[<asin>]
      * item_page → item_page, same asin, mutated options dict
                                            → click[<new option value>]
      * item_page → item_page, same asin, page suffix (Description/Features/
                    Reviews/Attributes)     → click[<tab name>]
      * item_page → search_results          → click[Back to Search]
      * item_page → done                    → click[Buy Now]

    Returns None for any transition that doesn't match a known pattern.
    Callers (trajectory_to_sft_examples) ABORT the whole trajectory in that
    case so the rendered prompt history can't desync (review item A7).
    """
    next_segs = _path_segments(next_url)
    prev_segs = _path_segments(prev_url)
    if not next_segs:
        return None
    page = next_segs[0]

    # done / buy
    if page == "done":
        return "click[Buy Now]"

    # search_results — could be initial search, pagination, or a re-search
    if page == "search_results" and len(next_segs) >= 4:
        next_query = _decode_query_list(next_segs[2])
        next_page_idx_raw = next_segs[3]
        try:
            next_page_idx = int(next_page_idx_raw)
        except ValueError:
            next_page_idx = 1
        if prev_segs and prev_segs[0] == "search_results" and len(prev_segs) >= 4:
            prev_query = _decode_query_list(prev_segs[2])
            try:
                prev_page_idx = int(prev_segs[3])
            except ValueError:
                prev_page_idx = 1
            if prev_query == next_query:
                if next_page_idx == prev_page_idx + 1:
                    return "click[Next >]"
                if next_page_idx == prev_page_idx - 1:
                    return "click[< Prev]"
                return None
            # Different query → fresh search.
            return f"search[{' '.join(next_query) if next_query else ''}]"
        if prev_segs and prev_segs[0] == "item_page":
            # On item page, navigated back to search results.
            return "click[Back to Search]"
        # From index (or unknown).
        return f"search[{' '.join(next_query) if next_query else ''}]"

    # item_page — could be ASIN click, option click, or tab click
    if page == "item_page" and len(next_segs) >= 3:
        next_asin = next_segs[2]
        # URL layout: /item_page/<task>/<asin>/<query>/<page>/<options>/<tab?>
        # → segs[3]=query, segs[4]=page, segs[5]=options, segs[6]=tab(optional)
        next_tab: str | None = None
        if len(next_segs) >= 7 and next_segs[6] in {
            "Description", "Features", "Reviews", "Attributes",
        }:
            next_tab = next_segs[6]
        next_options = next_segs[5] if len(next_segs) >= 6 else "%7B%7D"

        if prev_segs and prev_segs[0] == "item_page" and len(prev_segs) >= 3 and prev_segs[2] == next_asin:
            prev_tab: str | None = None
            if len(prev_segs) >= 7 and prev_segs[6] in {
                "Description", "Features", "Reviews", "Attributes",
            }:
                prev_tab = prev_segs[6]
            prev_options = prev_segs[5] if len(prev_segs) >= 6 else "%7B%7D"
            # Tab change → click[<tab>]
            if next_tab is not None and next_tab != prev_tab:
                return f"click[{next_tab}]"
            # Options changed → click[<new option value>]
            if next_options != prev_options:
                opt = _diff_options(prev_options, next_options)
                if opt is None:
                    return None
                return f"click[{opt}]"
            # Same asin, same options, same tab → not a meaningful action.
            return None
        # Coming from search_results or index → click on the listing.
        return f"click[{next_asin}]"

    return None


def _diff_options(prev_encoded: str, next_encoded: str) -> str | None:
    """Return the value of the option that newly appeared (or changed) when
    comparing two URL-encoded option dicts. Returns None on parse failure or
    when no single new value can be identified."""
    try:
        prev_d = ast.literal_eval(unquote(prev_encoded))
        next_d = ast.literal_eval(unquote(next_encoded))
    except Exception:
        return None
    if not isinstance(prev_d, dict) or not isinstance(next_d, dict):
        return None
    # New key, OR same key with a different value.
    new_pairs = [
        (k, v) for k, v in next_d.items()
        if k not in prev_d or prev_d.get(k) != v
    ]
    if len(new_pairs) != 1:
        return None
    return str(new_pairs[0][1])


# --------- Default ReAct prompt renderer ---------------------------------


def default_render_prompt(
    instruction: str,
    history: list[tuple[str, str]],   # list of (observation, action) for past turns
    current_observation: str,
) -> str:
    """Minimal ReAct-style prompt. The collector's renderer is more elaborate
    but for SFT we only need the contract: prompt ends right where the model
    should emit the next action. Override via the `render_prompt` arg of
    `trajectory_to_sft_examples`.
    """
    parts: list[str] = []
    parts.append(
        "You are an online shopping agent. Given the user instruction and "
        "the current page observation, output a single action."
    )
    parts.append("")
    parts.append(f"User instruction: {instruction}")
    parts.append("")
    for obs, act in history:
        parts.append(f"Observation: {obs.strip()}")
        parts.append(f"Action: {act}")
    parts.append(f"Observation: {current_observation.strip()}")
    parts.append("Action:")
    return "\n".join(parts)


# --------- Trajectory parsing --------------------------------------------


def _row_observation_text(row: dict[str, Any]) -> str:
    """Extract a human-readable observation string from a trajectory row.

    Falls back to a JSON dump of the `content` dict when no rendered
    observation is available.
    """
    content = row.get("content")
    if isinstance(content, dict):
        # WebShop typically stores the rendered text under content['text'] or
        # content['observation']; if neither, dump JSON keys for context.
        for key in ("observation", "text", "html_text", "rendered"):
            if key in content and isinstance(content[key], str):
                return content[key]
        return json.dumps({k: type(v).__name__ for k, v in content.items()})
    if isinstance(content, str):
        return content
    page = row.get("page", "?")
    return f"<page={page}>"


def _row_instruction(row: dict[str, Any]) -> str | None:
    goal = row.get("goal")
    if isinstance(goal, dict):
        return goal.get("instruction_text") or goal.get("query")
    return None


def trajectory_to_sft_examples(
    rows: list[dict[str, Any]],
    *,
    trajectory_id: str,
    render_prompt=default_render_prompt,
    min_reward: float = 0.0,
) -> list[SFTExample]:
    """Convert one trajectory's rows into SFT (prompt, action) pairs.

    Drops trajectories whose terminal `done` reward is below `min_reward`
    (default 0.0 = keep all). Set to e.g. 0.5 to filter for "succeeded".
    """
    if len(rows) < 2:
        return []
    instruction = _row_instruction(rows[0]) or ""
    final_reward = 0.0
    if isinstance(rows[-1], dict) and "reward" in rows[-1]:
        try:
            final_reward = float(rows[-1]["reward"])
        except (TypeError, ValueError):
            final_reward = 0.0
    if final_reward < min_reward:
        return []

    examples: list[SFTExample] = []
    history: list[tuple[str, str]] = []
    for i in range(len(rows) - 1):
        cur = rows[i]
        nxt = rows[i + 1]
        action = _action_from_url_transition(cur.get("url", ""), nxt.get("url", ""))
        if action is None:
            # Trajectory contains an un-recognised URL transition (e.g. an
            # action type the loader doesn't understand yet). Aborting the
            # WHOLE trajectory rather than skipping this single step keeps
            # the prompt history aligned with the action labels — skipping
            # would silently desync subsequent (obs, action) pairs and teach
            # mis-grounded supervision. (Review item A7.)
            return []
        obs_text = _row_observation_text(cur)
        prompt = render_prompt(instruction, list(history), obs_text)
        examples.append(
            SFTExample(
                prompt=prompt,
                action=action,
                instruction=instruction,
                step_idx=i,
                trajectory_id=trajectory_id,
                final_reward=final_reward,
            )
        )
        history.append((obs_text, action))
    return examples


# --------- Directory loader ----------------------------------------------


def load_jsonl_trajectory(path: str) -> list[dict[str, Any]]:
    """Load one .jsonl file into a list of row dicts. Skips malformed lines."""
    rows: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_sft_examples_from_directory(
    directory: str,
    *,
    render_prompt=default_render_prompt,
    min_reward: float = 0.0,
    max_files: int | None = None,
) -> list[SFTExample]:
    """Walk `directory` for *.jsonl trajectory files, produce SFTExamples.

    `min_reward=0.5` is a reasonable filter: keep only succeeded
    trajectories so the SFT teacher signal isn't polluted by failures.
    """
    files = sorted(f for f in os.listdir(directory) if f.endswith(".jsonl"))
    if max_files is not None:
        files = files[:max_files]
    out: list[SFTExample] = []
    for f in files:
        traj_id = f.removesuffix(".jsonl")
        rows = load_jsonl_trajectory(os.path.join(directory, f))
        out.extend(
            trajectory_to_sft_examples(
                rows,
                trajectory_id=traj_id,
                render_prompt=render_prompt,
                min_reward=min_reward,
            )
        )
    return out


def summarize_sft_dataset(examples: Iterable[SFTExample]) -> dict[str, Any]:
    """Quick stats for printing after a load."""
    examples = list(examples)
    n = len(examples)
    if n == 0:
        return {"n_examples": 0}
    by_action: dict[str, int] = {}
    rewards: list[float] = []
    for ex in examples:
        head = ex.action.split("[", 1)[0]
        by_action[head] = by_action.get(head, 0) + 1
        rewards.append(ex.final_reward)
    return {
        "n_examples": n,
        "n_trajectories": len({ex.trajectory_id for ex in examples}),
        "by_action_kind": by_action,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "reward_mean": sum(rewards) / n,
    }
