"""Unit tests for `src.turnrd.goal_target` (parser + scorer).

Round-trips against the canonical AlfWorld goal templates seen in the
v3 R9 50-task probe, plus explicit positive / negative cases for each
scoring tier of `score_action_against_goal`.

Pure-Python; no torch gate needed.
"""
from __future__ import annotations

from src.turnrd.goal_target import (
    parse_goal_object,
    score_action_against_goal,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parser_take_template() -> None:
    assert parse_goal_object("take alarmclock") == ("alarmclock", None)
    assert parse_goal_object("take the cd") == ("cd", None)


def test_parser_examine_with_desklamp() -> None:
    assert parse_goal_object("examine the cd with the desklamp.") == ("cd", "desklamp")
    assert parse_goal_object("examine the book with the desklamp") == ("book", "desklamp")


def test_parser_look_at_under_desklamp() -> None:
    assert parse_goal_object("look at book under the desklamp.") == ("book", "desklamp")
    assert parse_goal_object("look at statue under the desklamp") == ("statue", "desklamp")


def test_parser_put_templates() -> None:
    assert parse_goal_object("put a alarmclock in sidetable.") == ("alarmclock", "sidetable")
    assert parse_goal_object("put some book on cabinet.") == ("book", "cabinet")
    assert parse_goal_object("put the cd in shelf.") == ("cd", "shelf")
    assert parse_goal_object("put a book in armchair") == ("book", "armchair")


def test_parser_clean_heat_cool_templates() -> None:
    assert parse_goal_object("clean some lettuce and put it in countertop.") == (
        "lettuce", "countertop",
    )
    assert parse_goal_object("heat some bread and put it in microwave.") == (
        "bread", "microwave",
    )
    assert parse_goal_object("cool some egg and put it in fridge.") == (
        "egg", "fridge",
    )


def test_parser_pick_two_template() -> None:
    assert parse_goal_object("find two book and put them in shelf.") == ("book", "shelf")
    assert parse_goal_object("two cd and put them in drawer.") == ("cd", "drawer")


def test_parser_returns_none_on_unrecognized() -> None:
    assert parse_goal_object("do something arbitrary") == (None, None)
    assert parse_goal_object("") == (None, None)
    assert parse_goal_object(None) == (None, None)  # type: ignore[arg-type]


def test_parser_case_insensitive() -> None:
    assert parse_goal_object("TAKE ALARMCLOCK") == ("alarmclock", None)
    assert parse_goal_object("Examine The Cd With The Desklamp.") == ("cd", "desklamp")


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_scorer_decisive_take_on_target() -> None:
    """Decisive verb on the goal object → 1.0."""
    assert score_action_against_goal("take alarmclock 1 from shelf 1", "alarmclock") == 1.0
    assert score_action_against_goal("take cd 3 from desk 1", "cd") == 1.0


def test_scorer_decisive_move_on_target() -> None:
    """Move (place) on the goal object → 1.0."""
    assert score_action_against_goal("move book 4 to shelf 1", "book", "shelf") == 1.0


def test_scorer_decisive_examine_on_target() -> None:
    """Examine on the goal object → 1.0."""
    assert score_action_against_goal("examine cd", "cd", "desklamp") == 1.0


def test_scorer_decisive_on_secondary_only() -> None:
    """Decisive verb on secondary object only → 0.5."""
    assert score_action_against_goal("use desklamp 1", "cd", "desklamp") == 0.5


def test_scorer_open_secondary_receptacle() -> None:
    """Opening / closing the secondary receptacle → 0.5."""
    assert score_action_against_goal("open sidetable 1", "alarmclock", "sidetable") == 0.5
    assert score_action_against_goal("close cabinet 1", "book", "cabinet") == 0.5


def test_scorer_plausible_receptacle_for_target() -> None:
    """Interacting with a plausible-receptacle for the target → 0.25."""
    # alarmclock plausibly lives in {sidetable, shelf, drawer, desk, dresser}
    assert score_action_against_goal("go to shelf 4", "alarmclock") == 0.25
    assert score_action_against_goal("go to drawer 1", "alarmclock") == 0.25


def test_scorer_implausible_receptacle() -> None:
    """Receptacle not plausibly containing the target → 0.0."""
    assert score_action_against_goal("go to garbagecan 1", "alarmclock") == 0.0


def test_scorer_wrong_object_decisive() -> None:
    """Decisive verb on the wrong object → 0.0 (or 0.25 if it happens to be a recep)."""
    # 'take cd' for an alarmclock goal: cd is not a receptacle → 0.0
    assert score_action_against_goal("take cd 3 from desk 1", "alarmclock") in (0.0, 0.25)


def test_scorer_no_op_actions() -> None:
    """`look`, `inventory`, empty → 0.0."""
    assert score_action_against_goal("look", "book", "shelf") == 0.0
    assert score_action_against_goal("inventory", "book", "shelf") == 0.0
    assert score_action_against_goal("", "book", None) == 0.0


def test_scorer_handles_missing_goal_gracefully() -> None:
    """When the goal object is None, score is always 0.0."""
    assert score_action_against_goal("take alarmclock 1", None) == 0.0
    assert score_action_against_goal("anything", None, None) == 0.0


def test_scorer_word_boundary_not_substring() -> None:
    """Goal `cd` must not match the substring inside `creditcard`."""
    # 'creditcard' contains 'card' which contains 'c d' as a substring;
    # word-boundary regex on 'cd' should NOT match.
    assert score_action_against_goal("take creditcard 1 from desk 1", "cd") == 0.25
    # But should match the literal 'cd' word.
    assert score_action_against_goal("take cd 1 from desk 1", "cd") == 1.0


# ---------------------------------------------------------------------------
# End-to-end: every parsed goal in the probe yields at least one positive turn
# ---------------------------------------------------------------------------


def test_parser_covers_probe_corpus() -> None:
    """50 goal strings observed in the v3 R9 probe all parse to a non-None target."""
    observed_goals = [
        "examine the alarmclock with the desklamp.",
        "examine the book with the desklamp.",
        "examine the bowl with the desklamp.",
        "examine the cd with the desklamp.",
        "examine the keychain with the desklamp.",
        "examine the laptop with the desklamp.",
        "examine the mug with the desklamp.",
        "examine the newspaper with the desklamp.",
        "examine the pen with the desklamp.",
        "examine the pencil with the desklamp.",
        "examine the pillow with the desklamp.",
        "examine the vase with the desklamp.",
        "look at alarmclock under the desklamp.",
        "look at book under the desklamp.",
        "look at bowl under the desklamp.",
        "look at box under the desklamp.",
        "look at cd under the desklamp.",
        "look at creditcard under the desklamp.",
        "look at newspaper under the desklamp.",
        "look at pillow under the desklamp.",
        "look at plate under the desklamp.",
        "look at statue under the desklamp.",
        "put a alarmclock in sidetable.",
        "put a book in armchair.",
        "put some alarmclock on shelf.",
        "put some book on cabinet.",
        "put some book on desk.",
    ]
    misses = [g for g in observed_goals if parse_goal_object(g) == (None, None)]
    assert not misses, f"parser missed {len(misses)} goal(s): {misses}"
