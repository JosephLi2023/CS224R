from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


def _normalize_attr_token(s: Any) -> str:
    """Canonicalize an attribute token (lowercase + strip) for set membership."""
    return str(s).strip().lower()


def _extract_target_attrs(env: Any) -> set[str]:
    """Snapshot the target attribute set from the upstream WebShop env.

    Unions whatever goal paths are present (env.goal[...] or the older
    env.server.goals[session] fork). Returns the empty set when no goal
    payload is found; no introspection failure is fatal.
    """
    target: set[str] = set()
    if env is None:
        return target

    # Tier 1: env.goal (most common; set by reset(session=...)).
    goal: Any = getattr(env, "goal", None)
    if not isinstance(goal, dict):
        # Tier 2: env.server.goals[session_id] on older forks.
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
        # Category contributes one canonical token.
        category = goal.get("category")
        if isinstance(category, str) and category.strip():
            target.add(_normalize_attr_token(category))
        # Price upper-bound token, matching an "under $X" filter click.
        price = goal.get("price_upper")
        if isinstance(price, (int, float)):
            target.add(f"price_under_{int(price)}")
        # goal_options selector values; without these the attribute-progress
        # delta is zero on every option click (attributes never overlap them).
        goal_options = goal.get("goal_options")
        if isinstance(goal_options, dict):
            for v in goal_options.values():
                token = _normalize_attr_token(v)
                if token:
                    target.add(token)
    return target


def _extract_target_asin(env: Any) -> str | None:
    """Pull the target product's ASIN from the upstream env, if exposed.

    Tries env.goal["asin"] then env.goal["asins"][0]; returns None when
    neither is available. Used for the one-time +0.25 ASIN-landing bonus.
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

    Tries env.cur_options, env.clicked_options, then
    user_sessions[session]["options"] (on env and env.server) until one
    yields options; values are canonicalised. ASINs are excluded (handled
    by the +0.25 bonus). Always returns a set (empty -> no progress).
    """
    selected: set[str] = set()
    if env is None:
        return selected

    def _ingest(opts) -> bool:
        """Add normalized option values to `selected`; True iff non-empty."""
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

    # Tier 2: user_sessions[session]["options"] on env or env.server.
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
    """WebShop environment adapter with normalized state/action wiring.

    Upstream API: reset(...) -> obs or (obs, info); step(action_str) ->
    (obs, reward, done, info).

    With use_attribute_progress_intermediate_reward=True, reset() snapshots
    the target attrs/ASIN and step() emits a per-step attribute-progress
    delta (+ a one-time +0.25 ASIN-landing bonus) under
    info["intermediate_reward"]. Default False leaves runs unchanged.
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

        # Dense-signal opt-in. Default False preserves prior behavior.
        self._use_attr_progress_ir: bool = bool(use_attribute_progress_intermediate_reward)
        # Per-trajectory state (reset each reset()): target attrs/ASIN, the
        # prev target-overlap count, and a once-per-episode ASIN-bonus guard.
        self._target_attrs: set[str] | None = None
        self._target_asin: str | None = None
        self._prev_overlap: int = 0
        self._asin_bonus_fired: bool = False
        # Warn at most once per process on introspection failure.
        self._introspect_warning_emitted: bool = False

        self._env = self._build_webshop_env()

    def _build_webshop_env(self):
        """Build a WebShop env, trying web_agent_site.envs.web_agent_text_env
        then the legacy webshop.envs.web_agent_site_env."""
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

        # Legacy fallback for older WebShop forks under `webshop.`.
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
        """Reset the env. Maps collector's task_id=<int> -> WebShop's session
        kwarg (which seeds goal selection); absorbs other unknown kwargs."""
        self._steps = 0
        if "task_id" in kwargs:
            session = kwargs.pop("task_id")
            kwargs.setdefault("session", session)
        raw_obs, raw_info = self._normalize_reset(self._env.reset(**kwargs))

        # Snapshot target attrs + asin after reset(); skipped silently when
        # opt-in is off or attrs aren't exposed.
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
            # Prime prev-overlap so the first step's delta is well-defined.
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

        # Attribute-progress delta + one-time ASIN bonus, written to
        # info["intermediate_reward"]. Only when opt-in is on and target
        # attrs exist; the delta is clamped to >= 0, and the +0.25 ASIN
        # bonus fires at most once per episode.
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
