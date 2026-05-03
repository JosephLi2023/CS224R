from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ALFWorldState:
    observation_text: str
    valid_actions: list[str]
    step_index: int
    raw_observation: Any
    raw_info: dict[str, Any]


class ALFWorldAdapter:
    """
    ALFWorld environment adapter with normalized state/action wiring.

    This wrapper accepts several common ALFWorld API shapes and normalizes outputs.
    """

    def __init__(
        self,
        max_steps: int,
        task_split: str = "train",
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.max_steps = max_steps
        self.task_split = task_split
        self.env_kwargs = env_kwargs or {}

        self._steps = 0
        self._last_state: ALFWorldState | None = None
        self._env = self._build_alfworld_env()

    def _build_alfworld_env(self):
        import_error: Exception | None = None

        # Common ALFWorld text env path.
        try:
            from importlib import import_module

            module = import_module("alfworld.agents.environment")
            get_env = getattr(module, "get_environment", None)
            if callable(get_env):
                return get_env("alfred")(**self.env_kwargs)
        except Exception as exc:  # pragma: no cover - depends on local install
            import_error = exc

        # Fallback: some forks expose environment class directly.
        try:
            from importlib import import_module

            module = import_module("alfworld.agents.environment.alfred_tw_env")
            env_cls = getattr(module, "AlfredTWEnv")
            return env_cls(**self.env_kwargs)
        except Exception as exc:  # pragma: no cover - depends on local install
            import_error = exc

        raise ImportError(
            "Failed to import ALFWorld environment. Install ALFWorld and ensure "
            "its environment modules are importable. "
            f"Original error: {import_error}"
        )

    def _to_text(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        if isinstance(observation, list) and observation:
            return str(observation[0])
        if isinstance(observation, dict):
            for key in ("observation", "obs", "text", "state"):
                if key in observation:
                    return str(observation[key])
        return str(observation)

    def _extract_valid_actions(self, info: dict[str, Any] | None) -> list[str]:
        if not info:
            return []

        for key in (
            "admissible_commands",
            "admissible_actions",
            "valid_actions",
            "available_actions",
            "action_candidates",
        ):
            candidates = info.get(key)
            if isinstance(candidates, list):
                return [str(x) for x in candidates]

        return []

    def _normalize_reset(self, reset_out: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(reset_out, tuple):
            if len(reset_out) == 2 and isinstance(reset_out[1], dict):
                return reset_out[0], reset_out[1]
            if len(reset_out) >= 1:
                return reset_out[0], {}
        return reset_out, {}

    def _normalize_step(self, step_out: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        # Handle legacy and gymnasium-like variants.
        if isinstance(step_out, tuple):
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                return obs, float(reward), bool(done), info if isinstance(info, dict) else {"raw_info": info}
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated) or bool(truncated)
                return obs, float(reward), done, info if isinstance(info, dict) else {"raw_info": info}
        raise ValueError("Unexpected ALFWorld step output shape")

    def _make_state(self, observation: Any, info: dict[str, Any]) -> ALFWorldState:
        return ALFWorldState(
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
                "Pass a string command or ensure ALFWorld info exposes admissible commands."
            )
        if action < 0 or action >= len(valid_actions):
            raise IndexError(
                f"Action index {action} out of range for {len(valid_actions)} valid actions"
            )
        return valid_actions[action]

    def reset(self, **kwargs: Any) -> ALFWorldState:
        self._steps = 0

        # Try split-aware reset first where available.
        reset_kwargs = dict(kwargs)
        if "split" not in reset_kwargs:
            reset_kwargs["split"] = self.task_split

        try:
            raw_obs, raw_info = self._normalize_reset(self._env.reset(**reset_kwargs))
        except TypeError:
            raw_obs, raw_info = self._normalize_reset(self._env.reset())

        state = self._make_state(raw_obs, raw_info)
        self._last_state = state
        return state

    def step(self, action: str | int) -> tuple[ALFWorldState, float, bool, dict[str, Any]]:
        action_cmd = self._resolve_action(action)
        raw_obs, reward, done, info = self._normalize_step(self._env.step(action_cmd))

        self._steps += 1
        timeout = self._steps >= self.max_steps
        final_done = bool(done) or timeout
        info = dict(info)
        info["timeout"] = timeout
        info["resolved_action"] = action_cmd

        state = self._make_state(raw_obs, info)
        self._last_state = state
        return state, reward, final_done, info
