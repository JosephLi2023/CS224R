"""Unit tests for src.datasets.sft_alfworld (pure-Python; no torch / Modal)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.datasets.sft_alfworld import (
    SFTExample,
    _action_to_thought,
    _row_to_example,
    load_sft_examples_from_jsonl,
    summarize_sft_dataset,
    synthesize_sft_target,
)


# ---- Action → thought template -------------------------------------


def test_action_to_thought_navigation():
    out = _action_to_thought("go to fridge 1")
    assert "go" in out.lower()
    assert "fridge 1" in out


def test_action_to_thought_take_with_source():
    out = _action_to_thought("take apple 1 from countertop 2")
    assert "apple 1" in out
    assert "countertop 2" in out


def test_action_to_thought_put_in_on():
    out = _action_to_thought("put apple 1 in/on fridge 1")
    assert "apple 1" in out
    assert "fridge 1" in out


def test_action_to_thought_open_close_examine():
    assert "open" in _action_to_thought("open drawer 1").lower()
    assert "close" in _action_to_thought("close cabinet 1").lower()
    assert "examine" in _action_to_thought("examine countertop 1").lower()


def test_action_to_thought_look_inventory():
    assert "look" in _action_to_thought("look").lower()
    assert "carrying" in _action_to_thought("inventory").lower() or \
           "inventory" in _action_to_thought("inventory").lower()


def test_action_to_thought_use():
    out = _action_to_thought("use lamp 1")
    assert "lamp 1" in out


def test_action_to_thought_unknown_verb_falls_back():
    out = _action_to_thought("dance with apple")
    # Falls back to a generic "I'll <action>." statement; non-empty.
    assert out and isinstance(out, str)


# ---- synthesize_sft_target -----------------------------------------


def test_synthesize_sft_target_emits_react_block():
    out = synthesize_sft_target("go to fridge 1")
    assert out.startswith(" ")  # leading space concatenates with "Thought:"
    assert "Action: go to fridge 1" in out
    # Has both Thought sentence and Action line, separated by newline.
    parts = out.lstrip().split("\n")
    assert len(parts) == 2
    assert parts[1] == "Action: go to fridge 1"


def test_synthesize_sft_target_take_action():
    out = synthesize_sft_target("take apple 1 from countertop 2")
    assert "Action: take apple 1 from countertop 2" in out
    assert "apple 1" in out  # thought references the object


def test_synthesize_sft_target_inventory():
    out = synthesize_sft_target("inventory")
    assert "Action: inventory" in out


# ---- _row_to_example ------------------------------------------------


def test_row_to_example_minimal_valid_row():
    row = {
        "prompt": "Observation: hello\nThought:",
        "action": "look",
    }
    ex = _row_to_example(row)
    assert ex is not None
    assert ex.prompt == "Observation: hello\nThought:"
    assert ex.action == "look"
    # Defaults applied:
    assert ex.instruction == ""
    assert ex.step_idx == 0
    assert ex.trajectory_id == ""
    assert ex.final_reward == 0.0


def test_row_to_example_full_row():
    row = {
        "prompt": "P",
        "action": "go to fridge 1",
        "instruction": "put a hot apple in/on a countertop",
        "step_idx": 7,
        "trajectory_id": "trial-T20190907_174127_007128",
        "final_reward": 1.0,
    }
    ex = _row_to_example(row)
    assert ex is not None
    assert ex.step_idx == 7
    assert ex.final_reward == 1.0
    assert ex.instruction.startswith("put a hot apple")


def test_row_to_example_missing_required_returns_none():
    assert _row_to_example({"prompt": "P"}) is None
    assert _row_to_example({"action": "look"}) is None
    assert _row_to_example({"prompt": "", "action": "look"}) is None
    assert _row_to_example({"prompt": "P", "action": ""}) is None


def test_row_to_example_handles_bad_types():
    # Non-string prompt → None
    assert _row_to_example({"prompt": 42, "action": "look"}) is None
    # Bad numeric fields → coerced to safe defaults.
    ex = _row_to_example({
        "prompt": "P", "action": "look",
        "step_idx": "not-an-int", "final_reward": "not-a-float",
    })
    assert ex is not None
    assert ex.step_idx == 0
    assert ex.final_reward == 0.0


# ---- load_sft_examples_from_jsonl -----------------------------------


def _write_jsonl(tmp_path, rows: list[dict]) -> str:
    p = tmp_path / "trajs.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_load_synthetic_three_step_trajectory(tmp_path):
    rows = [
        {"prompt": "P0\nThought:", "action": "go to fridge 1",
         "instruction": "examine the apple", "step_idx": 0,
         "trajectory_id": "T01", "final_reward": 1.0},
        {"prompt": "P1\nThought:", "action": "open fridge 1",
         "instruction": "examine the apple", "step_idx": 1,
         "trajectory_id": "T01", "final_reward": 1.0},
        {"prompt": "P2\nThought:", "action": "examine apple 1",
         "instruction": "examine the apple", "step_idx": 2,
         "trajectory_id": "T01", "final_reward": 1.0},
    ]
    path = _write_jsonl(tmp_path, rows)
    examples = load_sft_examples_from_jsonl(path)
    assert len(examples) == 3
    assert examples[0].action == "go to fridge 1"
    assert examples[1].action == "open fridge 1"
    assert examples[2].action == "examine apple 1"
    assert all(e.trajectory_id == "T01" for e in examples)
    assert [e.step_idx for e in examples] == [0, 1, 2]


def test_load_min_reward_filter(tmp_path):
    rows = [
        {"prompt": "Pa", "action": "look", "trajectory_id": "T01",
         "final_reward": 1.0, "step_idx": 0},
        {"prompt": "Pb", "action": "look", "trajectory_id": "T02",
         "final_reward": 0.0, "step_idx": 0},
    ]
    path = _write_jsonl(tmp_path, rows)
    all_ex = load_sft_examples_from_jsonl(path, min_reward=0.0)
    assert len(all_ex) == 2
    only_succ = load_sft_examples_from_jsonl(path, min_reward=0.5)
    assert len(only_succ) == 1
    assert only_succ[0].trajectory_id == "T01"


def test_load_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_sft_examples_from_jsonl(str(p)) == []


def test_load_skips_malformed_lines(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"prompt": "P", "action": "look"}\n'
        "this is not json at all\n"
        "{partial json broken\n"
        '{"prompt": "P2", "action": "go to fridge 1"}\n'
    )
    examples = load_sft_examples_from_jsonl(str(p))
    assert len(examples) == 2
    assert examples[0].action == "look"
    assert examples[1].action == "go to fridge 1"


def test_load_max_examples_caps_count(tmp_path):
    rows = [
        {"prompt": f"P{i}", "action": "look", "trajectory_id": "T",
         "step_idx": i, "final_reward": 1.0}
        for i in range(10)
    ]
    path = _write_jsonl(tmp_path, rows)
    examples = load_sft_examples_from_jsonl(path, max_examples=3)
    assert len(examples) == 3


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sft_examples_from_jsonl(str(tmp_path / "nope.jsonl"))


# ---- summarize_sft_dataset ------------------------------------------


def test_summary_counts_actions_by_leading_verb(tmp_path):
    examples = [
        SFTExample("P", "go to fridge 1", "", 0, "T1", 1.0),
        SFTExample("P", "go to sink 1", "", 1, "T1", 1.0),
        SFTExample("P", "take apple 1 from countertop 2", "", 2, "T1", 1.0),
        SFTExample("P", "put apple 1 in/on fridge 1", "", 0, "T2", 0.5),
    ]
    s = summarize_sft_dataset(examples)
    assert s["n_examples"] == 4
    assert s["n_trajectories"] == 2
    assert s["by_action_kind"]["go"] == 2
    assert s["by_action_kind"]["take"] == 1
    assert s["by_action_kind"]["put"] == 1
    assert s["reward_min"] == 0.5
    assert s["reward_max"] == 1.0


def test_summary_empty_returns_zero():
    assert summarize_sft_dataset([]) == {"n_examples": 0}


# ---- Round-trip: runtime renderer + SFT target preserve `Thought:` boundary --


def test_runtime_renderer_concat_with_sft_target_is_clean():
    """The runtime ReAct renderer ends with 'Thought:' (no trailing
    space). `synthesize_sft_target` returns ' ...\\nAction: ...' (leading
    space). Concatenation must yield 'Thought: <thought>\\nAction: ...'
    — the same shape the model emits at runtime. A bug here is the same
    template-drift class that broke WebShop SFT v3.
    """
    from src.envs.prompts.react_alfworld import render_alfworld_turn_prompt

    state = SimpleNamespace(
        observation_text="You are in the middle of a kitchen. You see a fridge 1, a sink 1.",
        valid_actions=["go to fridge 1", "go to sink 1", "look", "inventory"],
        instruction="put a hot apple in/on a countertop",
    )
    prompt = render_alfworld_turn_prompt(state, [])
    assert prompt.rstrip().endswith("Thought:")
    target = synthesize_sft_target("go to fridge 1")
    full = prompt + target
    # The combined string must contain a properly-formatted Thought + Action block.
    assert "Thought: I should go to the fridge 1.\nAction: go to fridge 1" in full
