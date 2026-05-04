"""Unit tests for src.datasets.sft_webshop (pure-Python; no torch / Modal)."""

from __future__ import annotations

import json

from src.datasets.sft_webshop import (
    _action_from_url_transition,
    _decode_query_list,
    _path_segments,
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
    assert _action_from_url_transition(prev, nxt) == "click[buy]"


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
    assert ex[2].action == "click[buy]"


def test_synthetic_trajectory_preserves_instruction_and_reward():
    rows = _make_synthetic_trajectory()
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01")
    assert all(e.instruction == "find a red dress under $30" for e in ex)
    assert all(e.final_reward == 1.0 for e in ex)
    assert [e.step_idx for e in ex] == [0, 1, 2]


def test_prompt_accumulates_history_across_steps():
    rows = _make_synthetic_trajectory()
    ex = trajectory_to_sft_examples(rows, trajectory_id="T01")
    # step 0 (index) has NO prior history
    assert "Action: search" not in ex[0].prompt
    # step 1 (after search) has ONE prior action in history
    assert "Action: search[red dress]" in ex[1].prompt
    # step 2 (after click on item) has TWO prior actions
    assert "Action: search[red dress]" in ex[2].prompt
    assert "Action: click[B07AAAAAA]" in ex[2].prompt


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
