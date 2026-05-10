"""Unit tests for src.datasets.sft_webshop (pure-Python; no torch / Modal)."""

from __future__ import annotations

import json

from src.datasets.sft_webshop import (
    _action_from_url_transition,
    _decode_query_list,
    _path_segments,
    default_render_prompt,
    load_sft_examples_from_directory,
    summarize_sft_dataset,
    trajectory_to_sft_examples,
)


# ---- URL helpers ----------------------------------------------------


def test_path_segments_strips_host_and_empties():
    out = _path_segments("http://3.83.245.205:3000/search_results/abc/q/1")
    assert out == ["search_results", "abc", "q", "1"]


def test_decode_query_list_python_literal():
    encoded = "%5B%27red%27%2C%20%27dress%27%5D"  # ['red', 'dress']
    assert _decode_query_list(encoded) == ["red", "dress"]


def test_decode_query_list_falls_back_on_garbage():
    assert _decode_query_list("notalist") == ["notalist"]


# ---- URL → action inference ----------------------------------------


def test_action_search_from_index_to_search_results():
    prev = "http://host/task123"
    nxt = "http://host/search_results/task123/%5B%27red%27%2C%20%27dress%27%5D/1"
    assert _action_from_url_transition(prev, nxt) == "search[red dress]"


def test_action_click_asin_from_search_to_item_page():
    prev = "http://host/search_results/task1/foo/1"
    nxt = "http://host/item_page/task1/B0123ABCDE/%5B%27foo%27%5D/1/%7B%7D"
    assert _action_from_url_transition(prev, nxt) == "click[B0123ABCDE]"


def test_action_buy_when_transitioning_to_done():
    prev = "http://host/item_page/task1/B0/q/1/opts"
    nxt = "http://host/done/task1/B0/%7B%7D"
    assert _action_from_url_transition(prev, nxt) == "click[Buy Now]"


def test_action_none_for_unknown_transition():
    assert _action_from_url_transition("http://host/", "http://host/") is None


# ---- trajectory_to_sft_examples ------------------------------------


def _make_synthetic_trajectory() -> list[dict]:
    base = "http://host"
    task = "T01"
    return [
        {
            "page": "index",
            "url": f"{base}/{task}",
            "goal": {"instruction_text": "find a red dress under $30"},
            "content": {"observation": "Welcome to WebShop. Enter a search query."},
        },
        {
            "page": "search_results",
            "url": f"{base}/search_results/{task}/%5B%27red%27%2C%20%27dress%27%5D/1",
            "goal": {"instruction_text": "find a red dress under $30"},
            "content": {"observation": "Results: item-A item-B item-C"},
        },
        {
            "page": "item_page",
            "url": f"{base}/item_page/{task}/B07AAAAAA/%5B%27red%27%5D/1/%7B%7D",
            "goal": {"instruction_text": "find a red dress under $30"},
            "content": {"observation": "Red dress, $25"},
        },
        {
            "page": "done",
            "url": f"{base}/done/{task}/B07AAAAAA/%7B%7D",
            "goal": {"instruction_text": "find a red dress under $30"},
            "content": {"observation": "Purchased"},
            "reward": 1.0,
            "reward_info": {},
        },
    ]


def test_synthetic_trajectory_produces_3_examples():
    rows = _make_synthetic_trajectory()
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01")
    # 4 rows → 3 transitions → 3 (prompt, action) pairs
    assert len(ex) == 3
    assert ex[0].action == "search[red dress]"
    assert ex[1].action == "click[B07AAAAAA]"
    assert ex[2].action == "click[Buy Now]"


def test_synthetic_trajectory_preserves_instruction_and_reward():
    rows = _make_synthetic_trajectory()
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01")
    assert all(e.instruction == "find a red dress under $30" for e in ex)
    assert all(e.final_reward == 1.0 for e in ex)
    assert [e.step_idx for e in ex] == [0, 1, 2]


def test_prompt_accumulates_history_across_steps():
    rows = _make_synthetic_trajectory()
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01")
    # Count "Thought:" occurrences in each step's prompt.
    # The runtime renderer (which `default_render_prompt` now delegates to)
    # emits exactly TWO Thought references per prompt regardless of history
    # length: one in the system-prompt format-hint
    # (`  Thought: <one short reasoning sentence>`) and one trailing
    # `Thought:` token that conditions the model. Past turns are replayed
    # as `Observation:` + `Action:` ONLY (no synthesized Thought lines)
    # because the runtime collector parses Thoughts away after generation,
    # so the model never sees Thoughts in history at rollout time. Pinning
    # this count to 2 catches any future re-introduction of the v3-era
    # template drift that put synthesized Thoughts back in history.
    n_thought_step0 = ex[0].prompt.count("Thought:")
    n_thought_step1 = ex[1].prompt.count("Thought:")
    n_thought_step2 = ex[2].prompt.count("Thought:")
    assert n_thought_step0 == 2
    assert n_thought_step1 == 2
    assert n_thought_step2 == 2
    # History DOES still accumulate as (Observation, Action) pairs.
    assert "Action: search[red dress]" in ex[1].prompt
    assert "Action: click[B07AAAAAA]" in ex[2].prompt
    # And critically must NOT replay synthesized Thoughts in history.
    assert "Thought: I'll search for red dress." not in ex[1].prompt
    # All step prompts end with "Thought:" (matches runtime ReAct prefix).
    assert ex[0].prompt.rstrip().endswith("Thought:")
    assert ex[1].prompt.rstrip().endswith("Thought:")
    assert ex[2].prompt.rstrip().endswith("Thought:")


def test_min_reward_filter_drops_failed_trajectory():
    rows = _make_synthetic_trajectory()
    rows[-1]["reward"] = 0.2
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01", min_reward=0.5)
    assert ex == []


def test_min_reward_filter_keeps_high_reward():
    rows = _make_synthetic_trajectory()
    rows[-1]["reward"] = 1.0
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01", min_reward=0.5)
    assert len(ex) == 3


def test_short_trajectory_returns_empty():
    assert trajectory_to_sft_examples([], trajectory_id="X") == []
    assert trajectory_to_sft_examples([{"url": "u"}], trajectory_id="X") == []


# ---- load_sft_examples_from_directory ------------------------------


def test_load_from_directory_reads_jsonl_files(tmp_path):
    rows = [
        {"page": "index", "url": "http://h/T1", "goal": {"instruction_text": "buy X"}, "content": {"observation": "home"}},
        {"page": "search_results", "url": "http://h/search_results/T1/%5B%27x%27%5D/1", "goal": {"instruction_text": "buy X"}, "content": {"observation": "results"}},
        {"page": "item_page", "url": "http://h/item_page/T1/B0X/%5B%27x%27%5D/1/%7B%7D", "goal": {"instruction_text": "buy X"}, "content": {"observation": "item"}},
        {"page": "done", "url": "http://h/done/T1/B0X/%7B%7D", "goal": {"instruction_text": "buy X"}, "content": {"observation": "done"}, "reward": 1.0},
    ]
    p = tmp_path / "T1.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ex = load_sft_examples_from_directory(str(tmp_path))
    assert len(ex) == 3
    summary = summarize_sft_dataset(ex)
    assert summary["n_examples"] == 3
    assert summary["n_trajectories"] == 1
    assert summary["by_action_kind"] == {"search": 1, "click": 2}


# ---- A8: extended URL-action coverage ------------------------------


def test_action_pagination_next():
    prev = "http://h/search_results/T1/%5B%27red%27%5D/1"
    nxt  = "http://h/search_results/T1/%5B%27red%27%5D/2"
    assert _action_from_url_transition(prev, nxt) == "click[Next >]"


def test_action_pagination_prev():
    prev = "http://h/search_results/T1/%5B%27red%27%5D/3"
    nxt  = "http://h/search_results/T1/%5B%27red%27%5D/2"
    assert _action_from_url_transition(prev, nxt) == "click[< Prev]"


def test_action_re_search_different_query():
    prev = "http://h/search_results/T1/%5B%27red%27%5D/1"
    nxt  = "http://h/search_results/T1/%5B%27blue%27%2C%20%27dress%27%5D/1"
    assert _action_from_url_transition(prev, nxt) == "search[blue dress]"


def test_action_back_to_search():
    prev = "http://h/item_page/T1/B0X/q/1/%7B%7D"
    nxt  = "http://h/search_results/T1/%5B%27red%27%5D/1"
    assert _action_from_url_transition(prev, nxt) == "click[Back to Search]"


def test_action_tab_click_on_item_page():
    prev = "http://h/item_page/T1/B0X/q/1/%7B%7D"
    nxt  = "http://h/item_page/T1/B0X/q/1/%7B%7D/Description"
    assert _action_from_url_transition(prev, nxt) == "click[Description]"


def test_action_option_click_on_item_page():
    # prev: empty options dict ; next: {"size":"large"}
    prev = "http://h/item_page/T1/B0X/q/1/%7B%7D"
    nxt  = "http://h/item_page/T1/B0X/q/1/%7B%22size%22%3A%20%22large%22%7D"
    assert _action_from_url_transition(prev, nxt) == "click[large]"


def test_unknown_transition_returns_none():
    """e.g. same item_page same options → no meaningful action."""
    prev = "http://h/item_page/T1/B0X/q/1/%7B%7D"
    nxt  = "http://h/item_page/T1/B0X/q/1/%7B%7D"
    assert _action_from_url_transition(prev, nxt) is None


# ---- A7: abort trajectory on unrecognised transition ---------------


def test_unknown_transition_aborts_trajectory():
    """If any mid-trajectory transition can't be inferred, the WHOLE
    trajectory is dropped to prevent prompt-history desync."""
    rows = [
        {"page": "index", "url": "http://h/T1", "goal": {"instruction_text": "buy X"},
         "content": {"observation": "home"}},
        {"page": "search_results", "url": "http://h/search_results/T1/%5B%27x%27%5D/1",
         "goal": {"instruction_text": "buy X"}, "content": {"observation": "results"}},
        # Same page, same options → returns None → abort
        {"page": "search_results", "url": "http://h/search_results/T1/%5B%27x%27%5D/1",
         "goal": {"instruction_text": "buy X"}, "content": {"observation": "results"}},
        {"page": "done", "url": "http://h/done/T1/B0X/%7B%7D",
         "goal": {"instruction_text": "buy X"}, "content": {"observation": "done"}, "reward": 1.0},
    ]
    assert trajectory_to_sft_examples(rows, trajectory_id="T1") == []


# ---- Prompt-template alignment with runtime ReAct ------------------


def test_default_prompt_matches_runtime_template_structure():
    """The SFT loader's prompt template must end with 'Thought:' so the
    SFT model is conditioned on the SAME prefix the runtime collector uses.
    Diverging templates here was the root cause of the v3 R=0 result."""
    p = default_render_prompt(
        instruction="buy a red dress under $30",
        history=[],
        current_observation="Welcome to Amazon Shopping.",
    )
    assert p.rstrip().endswith("Thought:")
    # System prompt must mention the ReAct format the runtime expects.
    assert "ReAct" in p
    assert "Action: search[" in p
    assert "Action: click[" in p


def test_default_prompt_byte_identical_to_runtime_renderer():
    """`default_render_prompt` must produce a BYTE-IDENTICAL string to
    `render_webshop_turn_prompt` for the same inputs. This pins the
    SFT-↔-rollout prompt-parity contract and would catch any future
    re-introduction of template drift (e.g. someone tweaking the SFT
    system prompt without touching the runtime one).
    """
    from types import SimpleNamespace

    from src.envs.prompts.react_webshop import render_webshop_turn_prompt

    instruction = "buy a red dress under $30"
    history_pairs = [
        ("Welcome to Amazon Shopping.", "search[red dress]"),
        ("[Result 1] B0123ABCDE Red Maxi Dress $25", "click[B0123ABCDE]"),
    ]
    current_obs = "Red Maxi Dress page. Options: size [S, M, L]. Color: red."

    sft = default_render_prompt(
        instruction=instruction,
        history=history_pairs,
        current_observation=current_obs,
    )
    state = SimpleNamespace(
        observation_text=current_obs,
        instruction=instruction,
        valid_actions=[],
    )
    runtime_history = [
        SimpleNamespace(observation_text=o, action_text=a)
        for o, a in history_pairs
    ]
    runtime = render_webshop_turn_prompt(state, runtime_history)
    assert sft == runtime, (
        "SFT prompt drifted from runtime renderer.\n"
        f"SFT:\n{sft!r}\n\nRUNTIME:\n{runtime!r}"
    )


def test_default_prompt_truncates_long_history_like_runtime():
    """5-turn history must be truncated to the last 3 turns with the
    `... (N earlier turns omitted) ...` marker, matching the runtime
    renderer's `max_history_turns=3` default."""
    from types import SimpleNamespace

    from src.envs.prompts.react_webshop import render_webshop_turn_prompt

    history_pairs = [
        (f"obs_{i}", f"click[stub_{i}]") for i in range(5)
    ]
    p = default_render_prompt(
        instruction="buy stuff",
        history=history_pairs,
        current_observation="cur",
    )
    state = SimpleNamespace(
        observation_text="cur",
        instruction="buy stuff",
        valid_actions=[],
    )
    runtime_history = [
        SimpleNamespace(observation_text=o, action_text=a)
        for o, a in history_pairs
    ]
    rt = render_webshop_turn_prompt(state, runtime_history)
    assert p == rt
    assert "earlier turns omitted" in p, (
        "Long-history truncation marker missing; prompt parity broken."
    )
    # The two oldest turns must be dropped (only obs_2..obs_4 remain).
    assert "obs_0" not in p
    assert "obs_1" not in p
    assert "obs_4" in p


def test_synthesize_sft_target_emits_react_block():
    from src.datasets.sft_webshop import synthesize_sft_target
    out = synthesize_sft_target("search[red dress]")
    assert out.startswith(" ")  # leading space concatenates with prompt's "Thought:"
    assert "I'll search for red dress." in out
    assert "Action: search[red dress]" in out


def test_synthesize_sft_target_special_actions():
    from src.datasets.sft_webshop import synthesize_sft_target
    assert "Action: click[Buy Now]" in synthesize_sft_target("click[Buy Now]")
    assert "buy" in synthesize_sft_target("click[Buy Now]").lower()
    assert "next page" in synthesize_sft_target("click[Next >]").lower()
    assert "Action: click[B0123ABCDE]" in synthesize_sft_target("click[B0123ABCDE]")
    assert "Action: click[Description]" in synthesize_sft_target("click[Description]")
    assert "description" in synthesize_sft_target("click[Description]").lower()
