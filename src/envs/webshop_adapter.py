from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


def _normalize_attr_token(s: Any) -> str:
    """Canonicalize an attribute token for set membership.

    WebShop's `goal["attributes"]` is a list of free-text strings like
    `"black"`, `"size 6"`, `"under $50"`. We lowercase + strip so trivial
    whitespace/case differences against `clicked_options.values()` (which
    come from the rendered page text) don't desync the overlap count.
    """
    return str(s).strip().lower()


def _extract_target_attrs(env: Any) -> set[str]:
    """Snapshot the target attribute set from the upstream WebShop env.

    Tries the canonical attribute paths that the upstream `web_agent_site`
    package exposes on `WebAgentTextEnv` and its `server` backend:

      * `env.goal["attributes"]` — list[str] of attribute keywords.
      * `env.goal["category"]` — str category label.
      * `env.goal["query"]` — the user's free-text query (tokenised).
      * `env.goal["price_upper"]` — float price bucket (stringified).
      * `env.server.goals[env.session]["attributes" / ...]` — older forks.

    All keys are optional — we union whatever is present. Returns the
    empty set when no recognisable goal payload is found (the caller
    treats this as "signal unavailable" and degrades to legacy zero-IR
    behavior). Mirrors the AlfWorld `_extract_facts_set` defensiveness:
    no introspection failure should be fatal to env construction or
    step execution.
    """
    target: set[str] = set()
    if env is None:
        return target

    # Tier 1: `env.goal` (the most common upstream attr name on the
    # current task; mutated by `reset(session=...)`).
    goal: Any = getattr(env, "goal", None)
    if not isinstance(goal, dict):
        # Tier 2: `env.server.goals[session_id]` — older `web_agent_site`
        # forks store goals on a `server` sub-object indexed by session.
        server = getattr(env, "server", None)
        goals_dict = getattr(server, "goals", None) if server is not None else None
        session = getattr(env, "session", None)
        if isinstance(goals_dict, dict) and session is not None:
            goal = goals_dict.get(session)
        elif isinstance(goals_dict, list):
            try:
                goal = goals_dict[int(session)] if session is not None else None
            except (TypeError, ValueError, IndexError):
                goal = None

    if isinstance(goal, dict):
        attrs = goal.get("attributes")
        if isinstance(attrs, (list, tuple)):
            for a in attrs:
                token = _normalize_attr_token(a)
                if token:
                    target.add(token)
        # Category contributes one canonical token; the click that lands
        # in the category-matching listing earns credit too.
        category = goal.get("category")
        if isinstance(category, str) and category.strip():
            target.add(_normalize_attr_token(category))
        # Price upper-bound: stringify, normalise so a click on the
        # "under $X" filter (if surfaced by the upstream as an option
        # value) matches.
        price = goal.get("price_upper")
        if isinstance(price, (int, float)):
            target.add(f"price_under_{int(price)}")
        # `goal_options` — dict like {"color": "mint", "size": "7"}
        # mapping option NAME → exact VALUE the env's reward function
        # expects. These values are the labels of the item-page
        # selector buttons (`Select color: mint`, `Select size: 7`),
        # which when clicked land in `env.cur_options.values()`. Without
        # them in `target`, the attribute-progress delta is identically
        # zero on every option click (the description-tag tokens in
        # `goal.attributes` never overlap with the selector values in
        # `cur_options`), and the dense signal collapses to a 1-bit
        # ASIN-landing-bonus indicator — useless for TurnRDv2's V-head
        # which needs per-turn credit. Validated via
        # `infra/app_webshop_sft_gen.py::validate_dense_signal`:
        # without this line, mean IR on click_option = 0.0 across 39
        # captured option-click turns; with it, the delta fires
        # whenever the agent engages an option that's in the goal set.
        goal_options = goal.get("goal_options")
        if isinstance(goal_options, dict):
            for v in goal_options.values():
                token = _normalize_attr_token(v)
                if token:
                    target.add(token)
    return target


def _extract_target_asin(env: Any) -> str | None:
    """Pull the target product's ASIN from the upstream env, if exposed.

    Used by `WebShopAdapter.step()` to award a one-time +0.25 bonus when
    the agent's `click[<target_asin>]` lands on the goal product's item
    page — the WebShop analogue of the AlfWorld "subgoal pickup" turn
    that the facts-diff signal catches.

    Tries `env.goal["asin"]` then `env.goal["asins"][0]` (some forks
    track multiple acceptable ASINs). Returns None when neither is
    available; the caller treats None as "no asin bonus signal" and
    just emits the attribute-progress delta.
    """
    if env is None:
        return None
    goal: Any = getattr(env, "goal", None)
    if not isinstance(goal, dict):
        server = getattr(env, "server", None)
        goals_dict = getattr(server, "goals", None) if server is not None else None
        session = getattr(env, "session", None)
        if isinstance(goals_dict, dict) and session is not None:
            goal = goals_dict.get(session)
        elif isinstance(goals_dict, list):
            try:
                goal = goals_dict[int(session)] if session is not None else None
            except (TypeError, ValueError, IndexError):
                goal = None
    if not isinstance(goal, dict):
        return None
    asin = goal.get("asin")
    if isinstance(asin, str) and asin.strip():
        return asin.strip().lower()
    asins = goal.get("asins")
    if isinstance(asins, (list, tuple)) and asins and isinstance(asins[0], str):
        return asins[0].strip().lower()
    return None


def _extract_selected_attrs(env: Any) -> set[str]:
    """Snapshot the currently-selected attribute set from the upstream env.

    Used per-step to compute the attribute-progress delta against the
    target set. Sources, tried in order until one yields a non-empty dict:

      * `env.cur_options` — dict of {option_name: option_value} on
        forks that expose a flat env-level attr (rare in current
        princeton-nlp/WebShop).
      * `env.clicked_options` — alternative older-fork attr name.
      * `env.user_sessions[env.session]["options"]` — the canonical
        upstream path. The princeton-nlp/WebShop env stores
        per-session click state on `self.user_sessions[session_id]`,
        and option clicks populate the `"options"` sub-dict via
        `web_agent_text_env.py:410 session["options"][clickable_key] = clickable_name`.
        This is the path used by the env's own reward function to
        compute the terminal score, so reading from here keeps the
        IR signal consistent with the env's own evaluation.
      * `env.server.user_sessions[env.session]["options"]` — same
        attr but reached via the optional `server` wrapper (some
        forks split env vs server into two classes).

    Each value is canonicalised (lowercase + strip) and added. ASINs
    are deliberately NOT pulled in here — the +0.25 ASIN-landing
    bonus in `step()` handles that signal separately so we don't
    double-count.

    Always returns a set (possibly empty). Empty ⇒ no progress
    contribution this step (delta stays at 0), which keeps the
    fallback safe even when the upstream's option-tracking attr is
    absent. Pre-fix (the fork-incorrect `cur_options`-only path)
    `validate_dense_signal` showed mean_ir_by_action_kind[click_option]=0.0
    across 39 option-click turns; with the upstream `user_sessions[...]
    ["options"]` path added, the delta now fires whenever the agent
    clicks an option in the goal-options set. See the regression
    suite in tests/unit/test_webshop_adapter_attr_progress.py.
    """
    selected: set[str] = set()
    if env is None:
        return selected

    def _ingest(opts) -> bool:
        """Append normalized option values to `selected`; return True iff non-empty."""
        if not isinstance(opts, dict) or not opts:
            return False
        for v in opts.values():
            token = _normalize_attr_token(v)
            if token:
                selected.add(token)
        return True

    # Tier 1: flat env-level attrs (older / alt forks).
    for attr in ("cur_options", "clicked_options"):
        if _ingest(getattr(env, attr, None)):
            return selected

    # Tier 2: canonical upstream `user_sessions[session_id]["options"]`
    # path. Tried on both `env` directly and via `env.server` (fork-dependent).
    session_id = getattr(env, "session", None)
    if session_id is not None:
        for holder in (env, getattr(env, "server", None)):
            if holder is None:
                continue
            user_sessions = getattr(holder, "user_sessions", None)
            if not isinstance(user_sessions, dict):
                continue
            sess = user_sessions.get(session_id)
            if not isinstance(sess, dict):
                # Some forks key by str(session_id); try once more.
                sess = user_sessions.get(str(session_id))
            if not isinstance(sess, dict):
                continue
            if _ingest(sess.get("options")):
                return selected
    return selected


@dataclass
class WebShopState:
    observation_text: str
    valid_actions: list[str]
    step_index: int
    raw_observation: Any
    raw_info: dict[str, Any]


class WebShopAdapter:
    """
    WebShop environment adapter with normalized state/action wiring.

    Expected upstream API shape (common WebShop text env):
    - reset(...) -> obs OR (obs, info)
    - step(action_str) -> (obs, reward, done, info)

    This adapter normalizes observations and action candidates so trainers can
    consume a consistent interface.

    Dense-signal opt-in
    -------------------
    `use_attribute_progress_intermediate_reward=True` activates a per-step
    attribute-progress dense reward sourced from the upstream env's goal
    payload — the WebShop analogue of AlfWorld's `_use_facts_diff_ir`
    opt-in. When on:

      * `reset()` snapshots the target attribute set (color, size,
        category, price-bucket tokens) via `_extract_target_attrs(self._env)`
        and the target ASIN via `_extract_target_asin(self._env)`.
      * `step()` computes the per-step delta:
          `Δ = max(0, |curr ∩ target| − |prev ∩ target|) / |target|`
        — the fraction of target attributes newly engaged this turn.
      * `step()` adds a one-time +0.25 bonus on the turn that lands on
        the target ASIN's item page (mirrors AlfWorld's subgoal-pickup
        bonus pattern).
      * Writes the reconciled value into `info["intermediate_reward"]` +
        `info["intermediate_reward_source"] = "attr_progress"` — same
        keys the producer-side collector reads at
        `src/algorithms/grpo/collectors.py:251-252`.

    Default False preserves all existing WebShop runs byte-for-byte
    (no `intermediate_reward` key is added when the flag is off, so
    `progress_signal=None` continues to flow through the downstream
    dataset's per-batch gate exactly as before).
    """

    def __init__(
        self,
        max_steps: int,
        observation_mode: str = "text",
        task_split: str = "train",
        env_kwargs: dict[str, Any] | None = None,
        use_attribute_progress_intermediate_reward: bool = False,
    ) -> None:
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.task_split = task_split
        self.env_kwargs = env_kwargs or {}

        self._steps = 0
        self._last_state: WebShopState | None = None

        # Dense-signal opt-in. Default False preserves Phase 0 behavior
        # byte-for-byte (no `intermediate_reward` key emitted; producer's
        # per-trajectory gate keeps `progress_signal=None`).
        self._use_attr_progress_ir: bool = bool(use_attribute_progress_intermediate_reward)
        # Per-trajectory state (reset on every `reset()`):
        #   _target_attrs : set[str] — canonicalised target attribute
        #                   tokens snapshot at reset. None ⇒ signal
        #                   disabled OR introspection failed; step()
        #                   silently degrades to no-emit.
        #   _target_asin  : str|None — target product's ASIN (lowercased)
        #                   for the +0.25 landing bonus. None ⇒ no bonus.
        #   _prev_overlap : int — |prev_selected ∩ target| count carried
        #                   across step() calls so the delta is positive
        #                   only on the turn that newly engages a target
        #                   attribute.
        #   _asin_bonus_fired : bool — guards the +0.25 bonus to fire
        #                   at most once per episode (subsequent ASIN
        #                   page visits don't double-count).
        self._target_attrs: set[str] | None = None
        self._target_asin: str | None = None
        self._prev_overlap: int = 0
        self._asin_bonus_fired: bool = False
        # Sticky introspection-failure warning toggle so we don't spam
        # the orchestrator log: warn once per process if the upstream
        # env doesn't expose the goal attrs we expected.
        self._introspect_warning_emitted: bool = False

        self._env = self._build_webshop_env()

    def _build_webshop_env(self):
        """
        Build a WebShop env instance from common install layouts.

        Tries the canonical upstream package path `web_agent_site.envs.web_agent_text_env`,
        then the legacy alias `webshop.envs.web_agent_site_env` as a fallback.
        """
        import_error: Exception | None = None
        try:
            from importlib import import_module

            module = import_module("web_agent_site.envs.web_agent_text_env")
            env_cls = getattr(module, "WebAgentTextEnv")
            return env_cls(
                observation_mode=self.observation_mode,
                **self.env_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on install
            import_error = exc

        # Legacy fallback in case a user installs an older WebShop fork that
        # remapped the module tree under `webshop.`.
        try:
            from importlib import import_module

            module = import_module("webshop.envs.web_agent_site_env")
            env_cls = getattr(module, "WebAgentTextEnv")
            return env_cls(
                observation_mode=self.observation_mode,
                split=self.task_split,
                **self.env_kwargs,
            )
        except Exception as exc:  # pragma: no cover - depends on install
            import_error = exc

        raise ImportError(
            "Failed to import WebShop env. Install via Modal app infra/app_webshop_install.py "
            "or ensure `web_agent_site` is on PYTHONPATH. "
            f"Original error: {import_error}"
        )

    def _to_text(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        if isinstance(observation, dict):
            for key in ("observation", "obs", "text", "state"):
                if key in observation:
                    return str(observation[key])
        return str(observation)

    def _extract_valid_actions(self, info: dict[str, Any] | None) -> list[str]:
        if not info:
            return []

        for key in (
            "valid_actions",
            "available_actions",
            "admissible_actions",
            "action_candidates",
            "valid",
        ):
            candidates = info.get(key)
            if isinstance(candidates, list):
                return [str(x) for x in candidates]

        return []

    def _normalize_reset(self, reset_out: Any) -> tuple[Any, dict[str, Any]]:
        # Most modern envs return (obs, info), older variants may return obs only.
        if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
            return reset_out[0], reset_out[1]
        return reset_out, {}

    def _make_state(self, observation: Any, info: dict[str, Any]) -> WebShopState:
        return WebShopState(
            observation_text=self._to_text(observation),
            valid_actions=self._extract_valid_actions(info),
            step_index=self._steps,
            raw_observation=observation,
            raw_info=info,
        )

    def _resolve_action(self, action: str | int) -> str:
        if isinstance(action, str):
            return action

        if not isinstance(action, int):
            raise TypeError(f"Action must be str or int, got {type(action)}")

        valid_actions = self._last_state.valid_actions if self._last_state else []
        if not valid_actions:
            raise ValueError(
                "Integer action received but no valid action list is available. "
                "Pass a string action command or ensure env info exposes valid actions."
            )
        if action < 0 or action >= len(valid_actions):
            raise IndexError(
                f"Action index {action} out of range for {len(valid_actions)} valid actions"
            )
        return valid_actions[action]

    def reset(self, **kwargs: Any) -> WebShopState:
        """Reset the env. Maps collector's `task_id=<int>` → WebShop's `session`
        kwarg (which seeds goal selection); absorbs other unknown kwargs."""
        self._steps = 0
        if "task_id" in kwargs:
            session = kwargs.pop("task_id")
            kwargs.setdefault("session", session)
        raw_obs, raw_info = self._normalize_reset(self._env.reset(**kwargs))

        # Dense-signal: snapshot target attrs + asin AFTER reset() so
        # the upstream env's per-task goal selection has fired. When the
        # opt-in is off OR the upstream doesn't expose the attrs we
        # expect, fall through silently (intermediate_reward will not be
        # emitted ⇒ legacy `progress_signal=None` path).
        self._prev_overlap = 0
        self._asin_bonus_fired = False
        if self._use_attr_progress_ir:
            target = _extract_target_attrs(self._env)
            self._target_attrs = target if target else None
            self._target_asin = _extract_target_asin(self._env)
            if self._target_attrs is None and not self._introspect_warning_emitted:
                self._introspect_warning_emitted = True
                warnings.warn(
                    "use_attribute_progress_intermediate_reward=True but the "
                    "upstream WebShop env didn't expose a recognisable goal "
                    "attribute payload (tried env.goal[attributes/category/"
                    "price_upper] and env.server.goals[session]). The dense "
                    "signal will be silently disabled for this episode; "
                    "downstream V-head supervision falls back to the legacy "
                    "raw_env_reward signal.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            # Prime the previous-overlap counter with the initial-step
            # overlap so the first step's delta is well-defined.
            if self._target_attrs is not None:
                initial_selected = _extract_selected_attrs(self._env)
                self._prev_overlap = len(initial_selected & self._target_attrs)
        else:
            self._target_attrs = None
            self._target_asin = None

        state = self._make_state(raw_obs, raw_info)
        self._last_state = state
        return state

    def step(self, action: str | int) -> tuple[WebShopState, float, bool, dict[str, Any]]:
        action_cmd = self._resolve_action(action)

        out = self._env.step(action_cmd)
        if not (isinstance(out, tuple) and len(out) == 4):
            raise ValueError(
                "Unexpected WebShop step output. Expected (obs, reward, done, info)."
            )

        raw_obs, reward, done, info = out
        if not isinstance(info, dict):
            info = {"raw_info": info}

        self._steps += 1
        timeout = self._steps >= self.max_steps
        final_done = bool(done) or timeout
        info = dict(info)
        info["timeout"] = timeout
        info["resolved_action"] = action_cmd

        # Dense-signal: attribute-progress delta (+ one-time ASIN bonus).
        # Only fires when (a) the opt-in is on AND (b) target attrs were
        # discovered at reset(). Otherwise no `intermediate_reward` key
        # is added — Phase 0 byte-for-byte preserved.
        #
        # Two components, summed:
        #   1. Attribute progress: fraction of target attrs newly
        #      engaged this turn. Computed as
        #          max(0, |curr ∩ target| − prev_overlap) / |target|
        #      Negative deltas (an option click that REMOVES a previously-
        #      engaged option) are clamped to 0 so the V-head's
        #      non-negative shaping contract holds.
        #   2. One-time +0.25 bonus on the first turn where
        #      `resolved_action == click[<target_asin>]`. Fires at most
        #      once per episode (subsequent visits don't double-count).
        #      Mirrors the WebShop intuition that landing on the right
        #      item page is a meaningful subgoal that an attribute-only
        #      signal might miss (e.g. the target asin's listing happens
        #      to expose all its options pre-selected).
        if self._use_attr_progress_ir and self._target_attrs:
            curr_selected = _extract_selected_attrs(self._env)
            curr_overlap = len(curr_selected & self._target_attrs)
            attr_delta = max(0, curr_overlap - self._prev_overlap) / float(
                len(self._target_attrs)
            )
            self._prev_overlap = curr_overlap

            asin_bonus = 0.0
            if (
                self._target_asin is not None
                and not self._asin_bonus_fired
                and isinstance(action_cmd, str)
                and action_cmd.lower() == f"click[{self._target_asin}]"
            ):
                asin_bonus = 0.25
                self._asin_bonus_fired = True

            info["intermediate_reward"] = float(attr_delta + asin_bonus)
            info["intermediate_reward_source"] = "attr_progress"

        state = self._make_state(raw_obs, info)
        self._last_state = state
        return state, float(reward), final_done, info
