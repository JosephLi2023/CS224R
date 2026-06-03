from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


def _extract_expert_plan(info: Any) -> list[str]:
    """Pull the next-action list from the env info dict.

    Checks `extra.expert_plan` then `expert_plan`; returns the first
    non-empty list as list[str], or [] when no plan is available.
    """
    if not isinstance(info, dict):
        return []
    for key in ("extra.expert_plan", "expert_plan"):
        plan = info.get(key)
        if isinstance(plan, (list, tuple)) and plan:
            # Some adapters wrap once more in a per-batch list - peel it.
            first = plan[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return [str(x) for x in first]
            if isinstance(first, str):
                return [str(x) for x in plan]
    return []


def _extract_facts_set(info: Any) -> set[str]:
    """Pull the per-step PDDL fact base from info["facts"] as a set of
    stringified propositions for diff-friendly comparison.

    Canonicalizes via repr() since Proposition isn't always hashable, and
    tolerates the per-batch list wrapping. Returns the empty set when facts
    are missing or empty.
    """
    if not isinstance(info, dict):
        return set()
    facts = info.get("facts")
    if not isinstance(facts, (list, tuple)) or not facts:
        return set()
    # Peel the per-batch wrap (outer list holding the inner facts list).
    first = facts[0]
    if isinstance(first, (list, tuple)):
        facts = first
    if not facts:
        return set()
    return {repr(f) for f in facts}


@dataclass
class ALFWorldState:
    observation_text: str
    valid_actions: list[str]
    step_index: int
    raw_observation: Any
    raw_info: dict[str, Any]


class ALFWorldAdapter:
    """ALFWorld adapter normalizing common ALFWorld API shapes.

    task_split is injected into env_kwargs as the upstream `train_eval` key
    at construction. reset(task_id=N) maps N to game-file index
    N % len(game_files) for deterministic selection, else bare env.reset().
    """

    def __init__(
        self,
        max_steps: int,
        observation_mode: str = "text",
        task_split: str = "train",
        env_kwargs: dict[str, Any] | None = None,
        use_textworld_intermediate_reward: bool = False,
        use_facts_diff_intermediate_reward: bool = False,
    ) -> None:
        self.max_steps = max_steps
        self.observation_mode = observation_mode
        self.task_split = task_split
        # Inject task_split as train_eval without mutating the caller's dict.
        merged_kwargs: dict[str, Any] = dict(env_kwargs or {})
        merged_kwargs.setdefault("train_eval", task_split)
        self.env_kwargs = merged_kwargs

        # Prefer TextWorld's native intermediate_reward over the
        # expert-plan-length delta. Default False preserves prior behavior.
        self._use_tw_intermediate_reward: bool = bool(use_textworld_intermediate_reward)
        # True only when EnvInfos registration succeeds (set in _wrap_batch_env).
        self._tw_registration_succeeded: bool = False
        # Fail fast at construction if opt-in is on but textworld is missing.
        if self._use_tw_intermediate_reward and self._build_request_infos() is None:
            raise ImportError(
                "use_textworld_intermediate_reward=True but `textworld` is not "
                "importable. Install textworld or disable the flag."
            )

        self._steps = 0
        self._last_state: ALFWorldState | None = None
        # Most recent task_id-derived game index, for determinism checks.
        self._last_task_idx: int | None = None
        # Prev expert-plan length for the dense delta = max(0, prev - curr).
        self._prev_plan_len: int | None = None
        # PDDL-facts-diff opt-in: prefer the fact-set diff over the
        # plan-length delta (TextWorld still wins). Default False.
        self._use_facts_diff_ir: bool = bool(use_facts_diff_intermediate_reward)
        # Prev step's PDDL fact set. None when opt-in off or no facts.
        self._prev_facts_set: set[str] | None = None
        # Whether _env is TextWorld's batched wrapper (set in _wrap_batch_env).
        self._is_batched: bool = False
        self._env = self._build_alfworld_env()

    def _build_alfworld_env(self):
        import_error: Exception | None = None

        # AlfredTWEnv is the meta-env; .init_env(batch_size=1) gives a
        # gym-style BatchEnv. Forward observation_mode when accepted; fall
        # back to kwargs-only.
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

    def _build_request_infos(self) -> Any | None:
        """Build a TextWorld EnvInfos opting into native intermediate_reward,
        or None if textworld isn't importable.

        extras=["expert_plan"] and won/admissible_commands preserve keys that
        registration would otherwise strip.
        """
        try:
            from textworld import EnvInfos  # type: ignore[import-not-found]
        except ImportError:
            return None
        return EnvInfos(
            intermediate_reward=True,
            won=True,
            admissible_commands=True,
            extras=["expert_plan"],
        )

    def _wrap_batch_env(self, meta: Any) -> Any:
        """Convert AlfWorld's AlfredTWEnv meta-env into a gym-style env via
        .init_env(batch_size=1), stashing the meta as _alfred_meta and
        setting self._is_batched. Returns a gym-shaped env directly if
        .reset exists.

        EnvInfos registration (when opted in) tries init_env(request_infos=...),
        then meta.request_infos + init_env, then bare init_env with a warning.
        """
        # Already a gym-style env? Skip the wrap (some test fakes do this).
        if hasattr(meta, "reset") and hasattr(meta, "step"):
            self._is_batched = False
            return meta
        if not hasattr(meta, "init_env"):
            raise AttributeError(
                "ALFWorld env has neither `.reset()` nor `.init_env(batch_size=...)`; "
                "incompatible upstream API. Open an issue with the alfworld version."
            )

        env_infos = self._build_request_infos() if self._use_tw_intermediate_reward else None
        wrapped: Any = None
        if env_infos is not None:
            # Tier 1: kwarg.
            try:
                wrapped = meta.init_env(batch_size=1, request_infos=env_infos)
                self._tw_registration_succeeded = True
            except TypeError:
                # Tier 2: set request_infos as an attribute, then init_env.
                tier2_ok = False
                try:
                    setattr(meta, "request_infos", env_infos)
                except AttributeError:
                    pass
                else:
                    try:
                        wrapped = meta.init_env(batch_size=1)
                        self._tw_registration_succeeded = True
                        tier2_ok = True
                    except Exception:
                        # init_env failed after setattr; fall through to Tier 3.
                        pass
                if not tier2_ok:
                    # Tier 3: bare init; warn so smoke logs surface it.
                    warnings.warn(
                        "ALFWorld meta-env accepts neither `init_env(request_infos=...)` "
                        "nor `meta.request_infos = ...`; falling back to bare init. "
                        "TextWorld's native `intermediate_reward` field will NOT be "
                        "populated and the adapter will silently use the Phase 1 "
                        "expert-plan-length delta as the V-head signal.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    wrapped = meta.init_env(batch_size=1)
                    self._tw_registration_succeeded = False
        else:
            # Opt-in disabled: preserve prior behavior.
            wrapped = meta.init_env(batch_size=1)
            self._tw_registration_succeeded = False
        self._is_batched = True
        # Expose game_files for _select_task (the wrapper hides them).
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

        Unwraps the batch_size=1 list shape when self._is_batched.
        """
        if isinstance(reset_out, tuple):
            if len(reset_out) == 2 and isinstance(reset_out[1], dict):
                obs, info = reset_out
                if self._is_batched:
                    if isinstance(obs, (list, tuple)) and obs:
                        obs = obs[0]
                    info = self._unbatch_info(info)
                return obs, info
            if len(reset_out) >= 1:
                obs = reset_out[0]
                if self._is_batched and isinstance(obs, (list, tuple)) and obs:
                    obs = obs[0]
                return obs, {}
        return reset_out, {}

    def _normalize_step(self, step_out: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        """Normalize step output across legacy/gymnasium/batched API shapes.

        Unwraps the batch_size=1 per-field lists when self._is_batched.
        """
        if isinstance(step_out, tuple):
            if len(step_out) == 4:
                obs, reward, done, info = step_out
                if self._is_batched:
                    obs, reward, done, info = self._unbatch(obs, reward, done, info)
                return obs, float(reward), bool(done), info if isinstance(info, dict) else {"raw_info": info}
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                if self._is_batched:
                    obs, reward, terminated, info = self._unbatch(
                        obs, reward, terminated, info
                    )
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

        obs/reward/done are unwrapped from length-1 lists. info values are
        unwrapped only when a length-1 list/tuple wraps a list/tuple/dict;
        plain scalars pass through.
        """
        if isinstance(obs, (list, tuple)) and obs:
            obs = obs[0]
        if isinstance(reward, (list, tuple)) and reward:
            reward = reward[0]
        if isinstance(done, (list, tuple)) and done:
            done = done[0]
        info = self._unbatch_info(info)
        return obs, reward, done, info

    def _unbatch_info(self, info: Any) -> Any:
        """Unwrap batched info dicts. See `_unbatch` for the unwrap policy."""
        if not isinstance(info, dict):
            return info
        unbatched: dict[str, Any] = {}
        for k, v in info.items():
            if (
                isinstance(v, (list, tuple))
                and len(v) == 1
                and isinstance(v[0], (list, tuple, dict))
            ):
                unbatched[k] = v[0]
            else:
                unbatched[k] = v
        return unbatched

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
        """Return the env's game-files list (game_files/task_files), or [].

        Source of truth for deterministic task selection.
        """
        for attr in ("game_files", "task_files", "_game_files"):
            candidates = getattr(self._env, attr, None)
            if isinstance(candidates, (list, tuple)) and candidates:
                return list(candidates)
        return []

    def _select_task(self, task_id: int) -> int:
        """Map task_id -> game index (task_id % len(game_files), else task_id)
        and point the env at it via the first matching pointer attr."""
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
        """Reset the env. Maps task_id=<int> -> deterministic game-file
        selection (see _select_task); absorbs other unknown kwargs."""
        self._steps = 0
        task_id = kwargs.pop("task_id", None)
        if task_id is not None:
            self._last_task_idx = self._select_task(int(task_id))

        try:
            raw_obs, raw_info = self._normalize_reset(self._env.reset(**kwargs))
        except TypeError:
            raw_obs, raw_info = self._normalize_reset(self._env.reset())

        # Prime the expert-plan-length tracker; None -> step() yields delta 0.
        initial_plan = _extract_expert_plan(raw_info)
        self._prev_plan_len = len(initial_plan) if initial_plan else None

        # Prime the PDDL fact set; None -> step() yields delta 0 on step 1.
        if self._use_facts_diff_ir:
            initial_facts = _extract_facts_set(raw_info)
            self._prev_facts_set = initial_facts if initial_facts else None
        else:
            self._prev_facts_set = None

        state = self._make_state(raw_obs, raw_info)
        self._last_state = state
        return state

    def step(self, action: str | int) -> tuple[ALFWorldState, float, bool, dict[str, Any]]:
        action_cmd = self._resolve_action(action)
        # The batch env expects a length-1 command list; _normalize_step
        # unwraps the per-field batch lists on return.
        if hasattr(self._env, "_alfred_meta"):
            step_arg: Any = [action_cmd]
        else:
            step_arg = action_cmd
        try:
            raw_step_out = self._env.step(step_arg)
        except (AssertionError, TypeError):
            # Defensive: retry with a scalar if the env wants one.
            raw_step_out = self._env.step(action_cmd)
        raw_obs, reward, done, info = self._normalize_step(raw_step_out)

        self._steps += 1
        timeout = self._steps >= self.max_steps
        final_done = bool(done) or timeout
        info = dict(info)
        info["timeout"] = timeout
        info["resolved_action"] = action_cmd

        # Dense progress signal -> info["intermediate_reward"] (source in
        # info["intermediate_reward_source"]):
        #   1. (preferred) TextWorld's native intermediate_reward.
        #   2. (fallback) per-turn drop in the expert's remaining-plan length.
        # Always update _prev_plan_len so the fallback stays warm. Missing
        # plan or unprimed prev -> delta 0; max(0, .) avoids negative shaping.
        # Capture upstream BEFORE overwriting the key with the fallback.
        upstream_ir: Any = info.get("intermediate_reward") if (
            self._use_tw_intermediate_reward and self._tw_registration_succeeded
        ) else None
        # Peel the length-1 batch wrapping (e.g. [3]) for the type-check below.
        if isinstance(upstream_ir, (list, tuple)) and len(upstream_ir) == 1:
            upstream_ir = upstream_ir[0]
        curr_plan = _extract_expert_plan(info)
        curr_plan_len = len(curr_plan)
        prev_plan_len = (
            self._prev_plan_len if self._prev_plan_len is not None else curr_plan_len
        )
        fallback_delta = float(max(0, prev_plan_len - curr_plan_len))

        # PDDL-facts diff: max(0, len(new) - len(removed)). Only fires when
        # opt-in is on, prev set primed, and curr set non-empty.
        facts_delta: float | None = None
        curr_facts_set: set[str] | None = None
        if self._use_facts_diff_ir:
            curr_facts_set = _extract_facts_set(info)
            if curr_facts_set and self._prev_facts_set:
                new_facts = curr_facts_set - self._prev_facts_set
                removed_facts = self._prev_facts_set - curr_facts_set
                facts_delta = float(max(0, len(new_facts) - len(removed_facts)))
            # Always update the tracker (use curr when non-empty) so it stays
            # warm without diffing against a stale base.
            self._prev_facts_set = curr_facts_set if curr_facts_set else self._prev_facts_set

        if upstream_ir is not None and isinstance(upstream_ir, (int, float)) and not isinstance(upstream_ir, bool):
            # Clamp to non-negative; TextWorld's IR can go negative on reverts.
            info["intermediate_reward"] = max(0.0, float(upstream_ir))
            info["intermediate_reward_source"] = "textworld"
        elif facts_delta is not None:
            info["intermediate_reward"] = facts_delta
            info["intermediate_reward_source"] = "facts_diff"
        else:
            info["intermediate_reward"] = fallback_delta
            info["intermediate_reward_source"] = "expert_plan"
        self._prev_plan_len = curr_plan_len

        state = self._make_state(raw_obs, info)
        self._last_state = state
        return state, reward, final_done, info
