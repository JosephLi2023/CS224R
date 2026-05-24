"""Pure-Python goal-aware per-turn supervision target for TurnRDv2.

Plan: `turnrd_goal_aware_supervision`.

Given the AlfWorld goal substring (extracted by
`src.turnrd.goal_extractor.extract_goal_text`) and a single per-turn
action string, produce a per-turn scalar score in `[0, 1]` that the
V-head uses as its training target (blended with the existing
`progress_signal` per `train_turnrd`'s preference chain).

Two public functions:
- `parse_goal_object(goal_text)` → `(target_object, secondary_object)`
  parsed from the canonical AlfWorld goal templates.
- `score_action_against_goal(action_text, goal_obj, secondary_obj=None)`
  → `float in [0, 1]` reflecting how strongly the action engages the
  named goal object.

No torch, no LLM, no tokenizer — pure regex + string matching. Designed
to run in the producer hot path (one call per (trajectory, turn)) with
sub-millisecond overhead.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Canonical AlfWorld goal templates, in priority order. Each pattern is
# `(regex, capture_target_group, capture_secondary_group_or_None)`. The
# trailing period and any leading/trailing whitespace are tolerated by
# the regex; case-insensitive throughout.
#
# Templates observed in the v3 R9 50-task probe + plan-documented hard
# task types (clean/heat/cool/pick_two — not in probe but expected on
# the full eval pool).
_GOAL_TEMPLATES = [
    # look_at_obj_in_light: "look at X under the desklamp"
    (
        re.compile(
            r"look\s+at\s+(?:the\s+)?(?P<target>[a-z0-9_-]+)\s+under\s+the\s+(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # look_at_obj_in_light: "examine the X with the desklamp"
    (
        re.compile(
            r"examine\s+(?:the\s+)?(?P<target>[a-z0-9_-]+)\s+with\s+the\s+(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # pick_clean_then_place: "clean some X and put it in Y"
    (
        re.compile(
            r"clean\s+(?:some\s+|a\s+|the\s+)?(?P<target>[a-z0-9_-]+)\s+(?:and\s+put\s+(?:it|them)\s+)?(?:in|on)\s+(?:the\s+)?(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # pick_heat_then_place: "heat some X and put it in Y"
    (
        re.compile(
            r"heat\s+(?:some\s+|a\s+|the\s+)?(?P<target>[a-z0-9_-]+)\s+(?:and\s+put\s+(?:it|them)\s+)?(?:in|on)\s+(?:the\s+)?(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # pick_cool_then_place: "cool some X and put it in Y"
    (
        re.compile(
            r"cool\s+(?:some\s+|a\s+|the\s+)?(?P<target>[a-z0-9_-]+)\s+(?:and\s+put\s+(?:it|them)\s+)?(?:in|on)\s+(?:the\s+)?(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # pick_two_obj_and_place: "find two X and put them in Y"
    (
        re.compile(
            r"(?:find\s+)?two\s+(?P<target>[a-z0-9_-]+)\s+(?:and\s+put\s+(?:it|them)\s+)?(?:in|on)\s+(?:the\s+)?(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # pick_and_place_simple: "put a X in Y" / "put some X on Y" /
    # "put the X in Y" (the most generic template — keep LAST so the
    # more specific clean/heat/cool/pick_two templates match first).
    (
        re.compile(
            r"put\s+(?:a\s+|some\s+|the\s+)?(?P<target>[a-z0-9_-]+)\s+(?:in|on)\s+(?:the\s+)?(?P<sec>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", "sec",
    ),
    # Fallback: bare "take X" (no second object). Lower-priority than
    # the templates above so "put a X in Y" doesn't get matched as
    # `take` by accident.
    (
        re.compile(
            r"take\s+(?:a\s+|the\s+)?(?P<target>[a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        "target", None,
    ),
]


def parse_goal_object(goal_text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse the AlfWorld goal string into ``(target_object, secondary_object)``.

    Args:
        goal_text: the trimmed goal substring as returned by
            ``src.turnrd.goal_extractor.extract_goal_text`` — e.g.
            ``"take alarmclock"``, ``"put a book in cabinet"``,
            ``"examine the cd with the desklamp."``.

    Returns:
        A tuple ``(target, secondary)``:
        - ``target``: the named goal object (e.g. ``"alarmclock"``,
          ``"cd"``, ``"book"``), or ``None`` when no template matches.
        - ``secondary``: the secondary object — typically the receptacle
          the target should be placed in, or ``"desklamp"`` for the
          look_at_obj_in_light task type. ``None`` when no secondary
          slot exists for the matched template.
    """
    if not isinstance(goal_text, str) or not goal_text:
        return (None, None)
    cleaned = goal_text.strip().rstrip(".").strip()
    for regex, target_key, sec_key in _GOAL_TEMPLATES:
        m = regex.search(cleaned)
        if m:
            target = m.group(target_key).lower()
            secondary = (
                m.group(sec_key).lower() if sec_key is not None else None
            )
            return (target, secondary)
    return (None, None)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Decisive verbs from the v3 R9 credit-assignment analysis — when these
# verbs touch the goal object directly the turn carries near-all of the
# trajectory's credit. We give them the maximum score (1.0).
_DECISIVE_VERBS = frozenset(
    ["take", "examine", "use", "put", "move", "clean", "heat", "cool"]
)

# Cheap object→{plausible-receptacle-types} lookup for the +0.25 tier.
# Hand-built from AlfWorld's PDDL grammar; covers the receptacles
# observed in the v3 R9 probe. Permissive (a few false positives are
# preferable to false negatives — the +0.25 tier is a "near the goal"
# signal, not a "decisive turn" signal).
_OBJECT_TO_RECEPTACLES: dict[str, frozenset[str]] = {
    "alarmclock":  frozenset(["sidetable", "shelf", "drawer", "desk", "dresser"]),
    "book":        frozenset(["shelf", "drawer", "desk", "diningtable", "armchair", "sidetable", "sofa", "dresser", "cabinet"]),
    "bowl":        frozenset(["countertop", "cabinet", "sidetable", "diningtable", "fridge", "microwave"]),
    "box":         frozenset(["shelf", "drawer", "sidetable", "diningtable", "dresser", "desk"]),
    "bread":       frozenset(["countertop", "fridge", "microwave", "diningtable"]),
    "cd":          frozenset(["shelf", "drawer", "desk", "sidetable", "dresser"]),
    "creditcard":  frozenset(["sidetable", "drawer", "desk", "dresser", "shelf"]),
    "cup":         frozenset(["cabinet", "shelf", "countertop", "diningtable", "fridge"]),
    "egg":         frozenset(["fridge", "microwave", "countertop", "diningtable"]),
    "fork":        frozenset(["countertop", "cabinet", "drawer", "diningtable", "sink", "sinkbasin"]),
    "keychain":    frozenset(["sidetable", "drawer", "desk", "shelf", "dresser"]),
    "knife":       frozenset(["countertop", "cabinet", "drawer", "diningtable", "sink", "sinkbasin"]),
    "laptop":      frozenset(["desk", "sofa", "diningtable", "sidetable", "bed", "armchair"]),
    "lettuce":     frozenset(["countertop", "fridge", "diningtable"]),
    "mug":         frozenset(["sidetable", "diningtable", "shelf", "countertop", "cabinet", "desk", "coffeemachine"]),
    "newspaper":   frozenset(["sofa", "sidetable", "diningtable", "armchair", "dresser", "bed", "desk"]),
    "pen":         frozenset(["desk", "drawer", "sidetable", "shelf", "dresser"]),
    "pencil":      frozenset(["desk", "drawer", "sidetable", "shelf", "dresser"]),
    "pillow":      frozenset(["sofa", "armchair", "bed", "shelf"]),
    "plate":       frozenset(["countertop", "cabinet", "diningtable", "shelf", "fridge", "microwave"]),
    "potato":      frozenset(["countertop", "fridge", "microwave", "diningtable"]),
    "soapbar":     frozenset(["sinkbasin", "countertop", "shelf", "cart", "toilet"]),
    "spatula":     frozenset(["countertop", "drawer", "cabinet", "sink", "sinkbasin"]),
    "spoon":       frozenset(["countertop", "drawer", "cabinet", "sink", "sinkbasin", "diningtable"]),
    "statue":      frozenset(["shelf", "sidetable", "dresser", "desk", "diningtable"]),
    "tomato":      frozenset(["countertop", "fridge", "diningtable"]),
    "vase":        frozenset(["shelf", "sidetable", "dresser", "diningtable", "desk", "cabinet"]),
}

# Word-boundary cache for goal-object regex (slight perf win when the
# scorer is called once per turn across millions of turns).
_WORD_CACHE: dict[str, re.Pattern] = {}


def _word_re(word: str) -> re.Pattern:
    """Cache a `\\b<word>\\b` case-insensitive matcher."""
    key = word.lower()
    if key not in _WORD_CACHE:
        _WORD_CACHE[key] = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
    return _WORD_CACHE[key]


def score_action_against_goal(
    action_text: str,
    goal_obj: Optional[str],
    secondary_obj: Optional[str] = None,
) -> float:
    """Score a single per-turn action against the named goal object.

    Args:
        action_text: the trimmed per-turn action string (e.g.
            ``"take alarmclock 1 from shelf 1"``).
        goal_obj: the target object parsed from the goal (e.g.
            ``"alarmclock"``). Passing ``None`` makes the function
            return 0.0 (no goal-aware signal available).
        secondary_obj: the secondary slot from the goal — receptacle for
            place-tasks, ``"desklamp"`` for look_at_obj_in_light.

    Returns:
        A scalar in ``[0, 1]``:
        - **1.0** when a decisive verb (``take``, ``examine``, ``use``,
          ``put``, ``move``, ``clean``, ``heat``, ``cool``) touches the
          goal object directly.
        - **0.5** when the goal object is mentioned without a decisive
          verb (e.g. the goal_obj appears in the action text but no
          decisive verb does), OR when a decisive verb touches the
          *secondary* object (e.g. opening the target receptacle).
        - **0.25** when the action interacts (open/close/go to) with a
          receptacle type the goal object plausibly lives in.
        - **0.0** otherwise.

    Pure-string heuristic; tuned against the v3 R9 probe corpus.
    """
    if not isinstance(action_text, str) or not action_text or not goal_obj:
        return 0.0
    text = action_text.strip().lower()
    if not text:
        return 0.0

    # 1.0 — decisive verb on the goal object.
    has_goal_obj = bool(_word_re(goal_obj).search(text))
    if has_goal_obj:
        first_token = text.split(maxsplit=1)[0]
        if first_token in _DECISIVE_VERBS:
            return 1.0

    # 0.5 — secondary object engaged decisively (e.g. "use desklamp"
    # for look_at_obj_in_light, or "move book to shelf" for place tasks
    # where the secondary is the receptacle). Also fires when the goal
    # object is mentioned without a decisive verb.
    if secondary_obj:
        has_secondary = bool(_word_re(secondary_obj).search(text))
        if has_secondary:
            first_token = text.split(maxsplit=1)[0]
            if first_token in _DECISIVE_VERBS:
                return 0.5
            # Also count receptacle interactions like "open <secondary>",
            # "close <secondary>", "go to <secondary>".
            if first_token in {"open", "close"} or text.startswith("go to"):
                return 0.5
    if has_goal_obj:
        # Goal object mentioned without a decisive verb (e.g. "look",
        # "inventory" with no object engagement, or non-decisive verbs).
        return 0.5

    # 0.25 — interacting with a plausible-receptacle type for the
    # goal object (the "near the goal" tier).
    plausible = _OBJECT_TO_RECEPTACLES.get(goal_obj.lower(), frozenset())
    if plausible:
        for recep in plausible:
            if _word_re(recep).search(text):
                return 0.25

    # 0.0 — none of the above.
    return 0.0
