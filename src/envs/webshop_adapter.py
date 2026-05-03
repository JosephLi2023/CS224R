from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    """

    def __init__(
        self,
        max_steps: int,
        observation_mode: str = "text",
        task_split: str = "train",
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.task_split = task_split
        self.env_kwargs = env_kwargs or {}

        self._steps = 0
        self._last_state: WebShopState | None = None

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
        self._steps = 0
        raw_obs, raw_info = self._normalize_reset(self._env.reset(**kwargs))
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

        state = self._make_state(raw_obs, info)
        self._last_state = state
        return state, float(reward), final_done, info
