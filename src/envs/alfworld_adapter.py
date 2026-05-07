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

    Construction:
        - `observation_mode`: forwarded to the upstream env constructor when
          accepted (parallel to WebShop). The default `"text"` matches the
          ReAct-style prompt format the rollout collector expects.
        - `task_split`: ALFWorld's upstream `AlfredTWEnv` takes the split
          (`"train"` / `"eval_in_distribution"` / `"eval_out_of_distribution"`)
          at *construction* time, not at `reset()` time. We inject it into
          `env_kwargs` (under the `train_eval` key, the upstream's canonical
          name) when the caller hasn't already provided one. Configs that
          want a different upstream key should set `env_kwargs.train_eval`
          explicitly and leave `task_split` at its default — the `train_eval`
          value wins.

    Reset semantics:
        - `reset(task_id=N)` — when N is not None, the adapter maps it
          deterministically to a game-file index `N % len(game_files)` and
          points the underlying env at that game BEFORE calling `env.reset()`.
          This preserves H-GRPO's K-trajectories-per-task invariant: K
          parallel adapter instances all reset to task_id=N produce K
          rollouts on the SAME game.
        - When `task_id` is None or the adapter can't find a game-files
          list to index into, falls back to bare `env.reset()` (random
          game from the upstream's internal pointer).
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
        # Inject task_split as the upstream `train_eval` kwarg when the
        # caller hasn't already set one. Construct a fresh dict so we
        # don't mutate the caller's input.
        merged_kwargs: dict[str, Any] = dict(env_kwargs or {})
        merged_kwargs.setdefault("train_eval", task_split)
        self.env_kwargs = merged_kwargs

        self._steps = 0
        self._last_state: ALFWorldState | None = None
        # Records the most recent task_id-derived game index so callers
        # (and tests) can verify deterministic selection.
        self._last_task_idx: int | None = None
        self._env = self._build_alfworld_env()

    def _build_alfworld_env(self):
        import_error: Exception | None = None

        # AlfWorld's `AlfredTWEnv` is the META-env (loads game files,
        # holds config) — it does NOT expose `.reset()` / `.step()`
        # directly. To get a gym-style env we must call `.init_env(batch_size=1)`,
        # which returns a `BatchEnv` (TextWorld-gym wrapped) that
        # implements the standard env API. We do this once in the
        # adapter constructor so the `self._env` returned here is
        # immediately usable by `reset()` / `step()`.

        # Try the canonical ALFWorld text env path. Prefer to forward
        # `observation_mode` when the constructor accepts it (parallel to
        # WebShop); fall back to the kwargs-only construction for forks
        # that don't take it.
        try:
            from importlib import import_module

            module = import_module("alfworld.agents.environment")
            get_env = getattr(module, "get_environment", None)
            if callable(get_env):
                env_cls = get_env("alfred")
                try:
                    meta = env_cls(
                        observation_mode=self.observation_mode,
                        **self.env_kwargs,
                    )
                except TypeError:
                    meta = env_cls(**self.env_kwargs)
                return self._wrap_batch_env(meta)
        except Exception as exc:  # pragma: no cover - depends on local install
            import_error = exc

        # Fallback: some forks expose environment class directly.
        try:
            from importlib import import_module

            module = import_module("alfworld.agents.environment.alfred_tw_env")
            env_cls = getattr(module, "AlfredTWEnv")
            try:
                meta = env_cls(
                    observation_mode=self.observation_mode,
                    **self.env_kwargs,
                )
            except TypeError:
                meta = env_cls(**self.env_kwargs)
            return self._wrap_batch_env(meta)
        except Exception as exc:  # pragma: no cover - depends on local install
            import_error = exc

        raise ImportError(
            "Failed to import ALFWorld environment. Install ALFWorld and ensure "
            "its environment modules are importable. "
            f"Original error type: {type(import_error).__name__}; "
            f"Original error: {import_error!r}"
        )

    def _wrap_batch_env(self, meta: Any) -> Any:
        """Convert AlfWorld's `AlfredTWEnv` meta-env into a gym-style env.

        The meta-env exposes `.init_env(batch_size: int)` which returns a
        `BatchEnv` (TextWorld-gym wrapped) that implements `.reset()` /
        `.step()`. Some forks return the meta-env itself if it already
        looks gym-shaped — we only call `init_env(...)` when `.reset` is
        missing, so the adapter degrades gracefully.

        Also stashes the meta-env on the wrapper as `_alfred_meta` so
        `_select_task` can read the meta's `game_files` list (the wrapper
        typically doesn't expose it).
        """
        # Already a gym-style env? Skip the wrap (some test fakes do this).
        if hasattr(meta, "reset") and hasattr(meta, "step"):
            return meta
        if not hasattr(meta, "init_env"):
            raise AttributeError(
                "ALFWorld env has neither `.reset()` nor `.init_env(batch_size=...)`; "
                "incompatible upstream API. Open an issue with the alfworld version."
            )
        wrapped = meta.init_env(batch_size=1)
        # Surface the meta-env's `game_files` for `_select_task`'s
        # determinism check (the BatchEnv wrapper hides them).
        if not hasattr(wrapped, "game_files") and hasattr(meta, "game_files"):
            try:
                setattr(wrapped, "game_files", meta.game_files)
            except (AttributeError, TypeError):
                pass
        # Keep a back-pointer for callers that need the meta-env.
        try:
            setattr(wrapped, "_alfred_meta", meta)
        except (AttributeError, TypeError):
            pass
        return wrapped

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
        """Normalize reset output across legacy and batched API shapes.

        TextWorld batch env returns `(obs_list, info_dict_with_list_values)`.
        We unwrap for batch_size=1 so downstream code sees scalar obs +
        flat info dict.
        """
        if isinstance(reset_out, tuple):
            if len(reset_out) == 2 and isinstance(reset_out[1], dict):
                obs, info = reset_out
                if isinstance(obs, (list, tuple)) and obs:
                    obs = obs[0]
                if isinstance(info, dict):
                    info = {
                        k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v)
                        for k, v in info.items()
                    }
                return obs, info
            if len(reset_out) >= 1:
                obs = reset_out[0]
                if isinstance(obs, (list, tuple)) and obs:
                    obs = obs[0]
                return obs, {}
        return reset_out, {}

    def _normalize_step(self, step_out: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Normalize the step output across legacy/gymnasium/batched API shapes.

        TextWorld's batch env (returned by `AlfredTWEnv.init_env(batch_size=1)`)
        emits per-field LISTS of length `batch_size` (e.g. `obs=[str]`,
        `reward=[float]`, `done=[bool]`, `info={key: [val]}`). We unwrap
        for batch_size=1 so downstream code sees scalar reward/done and a
        flat info dict.
        """
        # Handle legacy and gymnasium-like variants.
        if isinstance(step_out, tuple):
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                obs, reward, done, info = self._unbatch(obs, reward, done, info)
                return obs, float(reward), bool(done), info if isinstance(info, dict) else {"raw_info": info}
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                obs, reward, terminated, info = self._unbatch(
                    obs, reward, terminated, info
                )
                # truncated may also be batched.
                if isinstance(truncated, (list, tuple)) and truncated:
                    truncated = truncated[0]
                done = bool(terminated) or bool(truncated)
                return obs, float(reward), done, info if isinstance(info, dict) else {"raw_info": info}
        raise ValueError("Unexpected ALFWorld step output shape")

    def _unbatch(
        self,
        obs: Any,
        reward: Any,
        done: Any,
        info: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Unwrap batch-size-1 returns from TextWorld's batched API.

        - `obs` = `[str]` → `str`
        - `reward` = `[float]` → `float`
        - `done` = `[bool]` → `bool`
        - `info` = `{k: [v]}` → `{k: v}` (only for keys whose value is a
          length-1 list/tuple; non-list fields pass through unchanged).
        """
        if isinstance(obs, (list, tuple)) and obs:
            obs = obs[0]
        if isinstance(reward, (list, tuple)) and reward:
            reward = reward[0]
        if isinstance(done, (list, tuple)) and done:
            done = done[0]
        if isinstance(info, dict):
            unbatched: dict[str, Any] = {}
            for k, v in info.items():
                if isinstance(v, (list, tuple)) and len(v) == 1:
                    unbatched[k] = v[0]
                else:
                    unbatched[k] = v
            info = unbatched
        return obs, reward, done, info

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

    def _game_files(self) -> list[Any]:
        """Return the underlying env's game-files list, if exposed.

        ALFWorld's `AlfredTWEnv` typically holds the loaded TextWorld game
        file paths under `game_files` (or `task_files` in some forks). This
        is the source of truth for deterministic task selection.
        """
        for attr in ("game_files", "task_files", "_game_files"):
            candidates = getattr(self._env, attr, None)
            if isinstance(candidates, (list, tuple)) and candidates:
                return list(candidates)
        return []

    def _select_task(self, task_id: int) -> int:
        """Map `task_id` → deterministic game index, then point env at it.

        Returns the resolved game index (`task_id % len(game_files)` when
        the game-files list is discoverable, else `task_id` verbatim — at
        minimum the `_last_task_idx` attribute records what was attempted
        so tests can assert determinism).

        ALFWorld's various forks expose the game pointer under different
        attribute names. We try the common ones; configs that need a
        different attribute can subclass and override `_select_task`.
        """
        game_files = self._game_files()
        idx = int(task_id) % len(game_files) if game_files else int(task_id)
        for attr in (
            "next_game_idx",
            "game_index",
            "_next_game",
            "_game_pointer",
        ):
            if hasattr(self._env, attr):
                setattr(self._env, attr, idx)
                break
        return idx

    def reset(self, **kwargs: Any) -> ALFWorldState:
        """Reset the env. Maps `task_id=<int>` → deterministic game-file
        selection (see `_select_task`); absorbs other unknown kwargs."""
        self._steps = 0
        task_id = kwargs.pop("task_id", None)
        if task_id is not None:
            self._last_task_idx = self._select_task(int(task_id))

        try:
            raw_obs, raw_info = self._normalize_reset(self._env.reset(**kwargs))
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
