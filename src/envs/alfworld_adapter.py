from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


def _extract_expert_plan(info: Any) -> list[str]:
    """Pull the next-action list from the env info dict.

    Checks `extra.expert_plan` then `expert_plan`, returning the first
    non-empty list as list[str], or [] when no plan is available (off-plan
    turn, terminal state, or fork that doesn't expose the expert). Kept in
    sync with infra/app_alfworld_sft_gen.py::_extract_expert_plan.
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

    TextWorld exposes the ground-truth PDDL facts as a
    list[textworld.Proposition]; diffing the per-step set gives a per-turn
    fluent-flip signal for V-head supervision. We canonicalize via repr()
    rather than hash() since Proposition is not hashable in some textworld
    releases. Tolerates the per-batch list-of-lists wrapping. Returns the
    empty set when facts are missing, non-list, or empty.
    """
    if not isinstance(info, dict):
        return set()
    facts = info.get("facts")
    if not isinstance(facts, (list, tuple)) or not facts:
        return set()
    # Per-batch wrap: outer list whose only element is the inner list of
    # propositions. Peel it.
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
    """ALFWorld environment adapter with normalized state/action wiring.

    Accepts several common ALFWorld API shapes and normalizes outputs.

    Construction:
        - observation_mode: forwarded to the upstream env constructor when
          accepted. Default "text" matches the ReAct-style prompt format.
        - task_split: AlfredTWEnv takes the split at construction time, not
          at reset() time. Injected into env_kwargs under the upstream's
          canonical `train_eval` key when the caller hasn't set one; an
          explicit env_kwargs.train_eval wins.

    Reset semantics:
        - reset(task_id=N): maps N deterministically to game-file index
          N % len(game_files) and points the env at it before env.reset(),
          preserving H-GRPO's K-trajectories-per-task invariant.
        - task_id None or no game-files list: falls back to bare
          env.reset() (random game from the upstream's pointer).
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
        # Inject task_split as the upstream `train_eval` kwarg when the
        # caller hasn't already set one. Construct a fresh dict so we
        # don't mutate the caller's input.
        merged_kwargs: dict[str, Any] = dict(env_kwargs or {})
        merged_kwargs.setdefault("train_eval", task_split)
        self.env_kwargs = merged_kwargs

        # Prefer TextWorld's native info["intermediate_reward"] (per-step
        # PDDL-fluent flips) over the synthesized expert-plan-length delta
        # as the V-head signal. Default False preserves prior behavior.
        self._use_tw_intermediate_reward: bool = bool(use_textworld_intermediate_reward)
        # Set True in _wrap_batch_env only when EnvInfos registration
        # succeeds; otherwise stays False -> step() uses the fallback.
        self._tw_registration_succeeded: bool = False
        # Fail fast at construction if opt-in is on but textworld is missing.
        if self._use_tw_intermediate_reward and self._build_request_infos() is None:
            raise ImportError(
                "use_textworld_intermediate_reward=True but `textworld` is not "
                "importable. Install textworld or disable the flag."
            )

        self._steps = 0
        self._last_state: ALFWorldState | None = None
        # Records the most recent task_id-derived game index so callers
        # (and tests) can verify deterministic selection.
        self._last_task_idx: int | None = None
        # Per-turn expert-plan length carried across step() calls to
        # compute the dense progress signal delta = max(0, prev - curr),
        # stashed on info["intermediate_reward"]. None -> no plan -> delta 0.
        self._prev_plan_len: int | None = None
        # PDDL-facts-diff opt-in. When True, step() computes the per-turn
        # fact-set diff and prefers it over the plan-length delta;
        # TextWorld upstream still wins when both are on. Default False.
        self._use_facts_diff_ir: bool = bool(use_facts_diff_intermediate_reward)
        # Previous step's PDDL fact set (stringified propositions). Reset
        # each reset(). None when opt-in is off or no facts exposed.
        self._prev_facts_set: set[str] | None = None
        # Whether _env is TextWorld's batched gym wrapper (set in
        # _wrap_batch_env). Test fakes skip the wrap, keeping this False.
        self._is_batched: bool = False
        self._env = self._build_alfworld_env()

    def _build_alfworld_env(self):
        import_error: Exception | None = None

        # AlfredTWEnv is the meta-env and does not expose .reset()/.step()
        # directly; .init_env(batch_size=1) returns a gym-style BatchEnv.

        # Canonical ALFWorld text env path. Forward observation_mode when
        # the constructor accepts it; fall back to kwargs-only for forks
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

    def _build_request_infos(self) -> Any | None:
        """Build a TextWorld EnvInfos registration object, or None if unavailable.

        Opts in to TextWorld's native per-step info["intermediate_reward"]
        at meta-env init time (AlfWorld won't populate it by default).
        extras=["expert_plan"] whitelists the fallback key so it survives
        registration; won/admissible_commands preserve other
        downstream-relied-on keys. Returns None when textworld isn't
        importable so callers can raise loudly or no-op.
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
        """Convert AlfWorld's AlfredTWEnv meta-env into a gym-style env.

        The meta-env's .init_env(batch_size=1) returns a gym-style BatchEnv.
        Some forks return a gym-shaped env directly, so we only call
        init_env(...) when .reset is missing. Stashes the meta-env as
        _alfred_meta (for _select_task's game_files) and sets
        self._is_batched = True so _normalize_* helpers unwrap the
        per-batch shape.

        EnvInfos registration (when _use_tw_intermediate_reward) uses a
        3-tier fallback since forks differ on the request_infos kwarg:
          - Tier 1: init_env(batch_size=1, request_infos=env_infos).
          - Tier 2: set meta.request_infos, then bare init_env(batch_size=1).
          - Tier 3: bare init_env + warn; _tw_registration_succeeded stays
            False so step() uses the fallback.
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
                # Fallback: set request_infos as an attribute, then call init_env.
                # AttributeError means this fallback is unsupported; init-time
                # failures are handled by the final fallback below.
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
                        # init_env failed even after setattr succeeded;
                        # fall through to Tier 3 rather than propagating.
                        pass
                if not tier2_ok:
                    # Tier 3: silent-fallback. Warn loudly so smoke logs
                    # surface the regression.
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

        TextWorld's batch env returns (obs_list, info_dict_with_list_values);
        we unwrap for batch_size=1 so downstream sees scalar obs + flat info.
        Only fires when self._is_batched; raw shape passes through otherwise.
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
        """Normalize the step output across legacy/gymnasium/batched API shapes.

        TextWorld's batch env emits per-field lists of length batch_size; we
        unwrap for batch_size=1 so downstream sees scalar reward/done and a
        flat info dict. Only fires when self._is_batched.
        """
        # Handle legacy and gymnasium-like variants.
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

        - obs = [str] -> str
        - reward = [float] -> float
        - done = [bool] -> bool
        - info = {k: [v]} -> {k: v}, but only when the value is a length-1
          list/tuple whose element is itself a list/tuple/dict (the
          TextWorld batch shape). Plain scalar info values pass through.
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
        """Map task_id -> deterministic game index, then point env at it.

        Returns task_id % len(game_files) when the list is discoverable,
        else task_id verbatim. Forks expose the game pointer under
        different attr names; we try the common ones.
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

        # Prime the expert-plan-length tracker for the dense signal in
        # step(). No expert_plan -> tracker None -> step() yields delta 0.
        initial_plan = _extract_expert_plan(raw_info)
        self._prev_plan_len = len(initial_plan) if initial_plan else None

        # Prime the per-step PDDL fact set. None when opt-in is off or no
        # facts exposed, so step() yields delta 0 instead of rewarding
        # "everything new" on the first step.
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
        # TextWorld's batch env (the AlfredTWEnv.init_env(batch_size=1)
        # wrapper) expects a LIST of commands of length=batch_size. With
        # batch_size=1 we send a single-element list; `_normalize_step`
        # unwraps the per-field batch lists on the way back.
        if hasattr(self._env, "_alfred_meta"):
            step_arg: Any = [action_cmd]
        else:
            step_arg = action_cmd
        try:
            raw_step_out = self._env.step(step_arg)
        except (AssertionError, TypeError):
            # Defensive: if the env actually wants a scalar (e.g. test
            # fakes), retry without the wrap.
            raw_step_out = self._env.step(action_cmd)
        raw_obs, reward, done, info = self._normalize_step(raw_step_out)

        self._steps += 1
        timeout = self._steps >= self.max_steps
        final_done = bool(done) or timeout
        info = dict(info)
        info["timeout"] = timeout
        info["resolved_action"] = action_cmd

        # Dense progress shaping signal, reconciled from two sources:
        #   1. (preferred) TextWorld's native info["intermediate_reward"]:
        #      newly-satisfied PDDL fluents minus reverted ones per step.
        #   2. (fallback) per-turn reduction in the handcoded expert's
        #      remaining-plan length.
        # Written to info["intermediate_reward"] (never overwrites
        # raw_env_reward); the chosen source is recorded under
        # info["intermediate_reward_source"]. Always recompute and update
        # _prev_plan_len so the fallback stays warm.
        #
        # Edge cases (fallback path):
        #   - Plan missing/empty mid-trajectory -> curr_plan_len 0, emit 0;
        #     the max(0, .) clamp also prevents negative shaping on re-plan.
        #   - _prev_plan_len never primed -> fall back to curr_plan_len so
        #     delta 0 instead of rewarding the first step.
        # Capture upstream BEFORE we overwrite the key with the fallback.
        upstream_ir: Any = info.get("intermediate_reward") if (
            self._use_tw_intermediate_reward and self._tw_registration_succeeded
        ) else None
        # In the batched path, scalar info values arrive wrapped as
        # length-1 lists (e.g. [3]); _unbatch_info leaves these unchanged,
        # so peel the per-batch wrapping here for the type-check below.
        if isinstance(upstream_ir, (list, tuple)) and len(upstream_ir) == 1:
            upstream_ir = upstream_ir[0]
        curr_plan = _extract_expert_plan(info)
        curr_plan_len = len(curr_plan)
        prev_plan_len = (
            self._prev_plan_len if self._prev_plan_len is not None else curr_plan_len
        )
        fallback_delta = float(max(0, prev_plan_len - curr_plan_len))

        # PDDL-facts diff: net delta = max(0, len(new) - len(removed)) so
        # satisfying one fluent while reverting another scores 0. Only
        # fires when opt-in is on, prev set was primed, and curr set is
        # non-empty; otherwise stays None.
        facts_delta: float | None = None
        curr_facts_set: set[str] | None = None
        if self._use_facts_diff_ir:
            curr_facts_set = _extract_facts_set(info)
            if curr_facts_set and self._prev_facts_set:
                new_facts = curr_facts_set - self._prev_facts_set
                removed_facts = self._prev_facts_set - curr_facts_set
                facts_delta = float(max(0, len(new_facts) - len(removed_facts)))
            # Always update tracker so it stays warm even if this step
            # didn't emit (e.g. first step after reset). Use the curr
            # set even when empty so a subsequent populated step
            # doesn't spuriously diff against a stale fact base.
            self._prev_facts_set = curr_facts_set if curr_facts_set else self._prev_facts_set

        if upstream_ir is not None and isinstance(upstream_ir, (int, float)) and not isinstance(upstream_ir, bool):
            # max(0, .) clamp keeps shaping non-negative; TextWorld's
            # intermediate_reward can be negative when fluents are reverted.
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
