from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any


def _extract_expert_plan(info: Any) -> list[str]:
    """Pull the next-action list from the env info dict.

    AlfWorld's handcoded expert exposes its remaining-action plan under
    several possible keys depending on the upstream version / wrapper
    layer:
      - `extra.expert_plan` (TextWorld batched-info convention)
      - `expert_plan`        (some non-batched wrappers)

    Returns the first non-empty list found, normalized to ``list[str]``.
    Returns ``[]`` when no plan is available (off-plan turn, terminal
    state, or fork that doesn't expose the expert).

    NOTE: ported from ``infra/app_alfworld_sft_gen.py::_extract_expert_plan``
    so the two consumers (SFT generator + per-turn shaping signal) stay
    on the same edge-case handling. Keep them in sync if you modify the
    canonical key list.
    """
    if not isinstance(info, dict):
        return []
    for key in ("extra.expert_plan", "expert_plan"):
        plan = info.get(key)
        if isinstance(plan, (list, tuple)) and plan:
            # Some adapters wrap once more in a per-batch list — peel it.
            first = plan[0]
            if isinstance(first, (list, tuple)) and first and isinstance(first[0], str):
                return [str(x) for x in first]
            if isinstance(first, str):
                return [str(x) for x in plan]
    return []


def _extract_facts_set(info: Any) -> set[str]:
    """Pull the per-step PDDL fact base from the env info dict as a set
    of stringified propositions for diff-friendly comparison.

    AlfWorld's TextWorld backend exposes the full ground-truth PDDL fact
    base for the current world state under ``info["facts"]`` as a
    ``list[textworld.Proposition]`` (length ~147 for a typical kitchen
    task). Diffing the per-step set gives us a per-turn
    PDDL-fluent-flip signal that's a stronger V-head supervision target
    than the (broken) plan-length delta or the (mis-populated) native
    ``intermediate_reward`` field.\n\n    We canonicalize each Proposition with ``repr()`` rather than
    relying on ``hash()``: ``textworld.Proposition`` carries a tuple of
    ``Variable`` objects, and in some textworld releases these are not
    hashable directly (``set()`` would raise ``TypeError``). ``repr()``
    yields stable, comparable strings immune to upstream changes.

    Mirrors ``_extract_expert_plan``'s tolerance for the per-batch
    list-of-lists wrapping that TextWorld's batched env emits (a
    length-1 outer list whose only element is the actual list of
    Propositions).

    Returns the empty set when the key is missing, the value is not a
    list/tuple, or the list is empty.
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

        # Phase 2 opt-in: prefer TextWorld's native `info["intermediate_reward"]`
        # (per-step PDDL-fluent flips) over Phase 1's synthesized
        # expert-plan-length delta as the V-head supervision source.
        # Default False ensures Phase 1 behavior is byte-for-byte
        # preserved when the flag is absent.
        self._use_tw_intermediate_reward: bool = bool(use_textworld_intermediate_reward)
        # Set True in `_wrap_batch_env` only when EnvInfos registration
        # via Tier 1 (kwarg) or Tier 2 (attribute) succeeds. Tier 3
        # silent-fallback or opt-in disabled ⇒ stays False ⇒ `step()`
        # routes the Phase 1 fallback regardless.
        self._tw_registration_succeeded: bool = False
        # Loud failure on missing `textworld` when opt-in is on — fail
        # fast at construction time rather than silently at step time.
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
        # Per-turn expert-plan length carried across `step()` calls so
        # we can compute the dense progress shaping signal
        # `Δ = max(0, prev_plan_len - curr_plan_len)` and stash it on
        # `info["intermediate_reward"]`. Reset to None on each `reset()`,
        # then primed from the initial info dict (when an expert plan is
        # exposed). None ⇒ no plan available ⇒ shaping delta defaults to
        # 0 (off-plan / non-expert envs are non-disruptive).
        self._prev_plan_len: int | None = None
        # Phase-3 (PDDL-facts diff) opt-in. When True, `step()` will
        # additionally compute the per-turn fact-set diff against the
        # previous step and prefer it over the (broken-in-prod) Phase 1
        # plan-length delta. TextWorld upstream still wins when both
        # flags are on. Default False ⇒ Phase 1 byte-for-byte behavior.
        self._use_facts_diff_ir: bool = bool(use_facts_diff_intermediate_reward)
        # Per-instance state: previous step's PDDL fact set as a set of
        # stringified propositions. Reset on every `reset()`. Stays None
        # when opt-in is off OR no `info["facts"]` is exposed by the
        # env, which keeps the Phase-3 path inert.
        self._prev_facts_set: set[str] | None = None
        # Whether `_env` is TextWorld's batched gym wrapper (set True in
        # `_wrap_batch_env` when we call `init_env(batch_size=1)`). Test
        # fakes that override `_build_alfworld_env` skip the wrap so this
        # default keeps unbatch behavior off for them.
        self._is_batched: bool = False
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

    def _build_request_infos(self) -> Any | None:
        """Build a TextWorld `EnvInfos` registration object, or None if unavailable.

        Phase 2 surfaces TextWorld's native per-step `info["intermediate_reward"]`
        (count of newly-satisfied PDDL fluents minus reverted ones) as
        the preferred V-head supervision source on ALFWorld. To do so we
        must opt-in via `EnvInfos(intermediate_reward=True, ...)` at
        meta-env init time — AlfWorld will not populate the field by
        default.

        Defensive `extras=["expert_plan"]`: registering an EnvInfos
        object can strip any incidentally-populated keys that aren't
        listed. Phase 1's fallback source (`extra.expert_plan` /
        `expert_plan`) is incidentally populated today; we whitelist it
        explicitly so the fallback path stays warm even when Tier 1
        registration succeeds.

        `won=True` and `admissible_commands=True` similarly preserve
        downstream-relied-on keys (the latter feeds
        `_extract_valid_actions`).

        Returns None when `textworld` isn't importable so callers can
        choose to raise loudly (opt-in on) or silently no-op (opt-in off).
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
        """Convert AlfWorld's `AlfredTWEnv` meta-env into a gym-style env.

        The meta-env exposes `.init_env(batch_size: int)` which returns a
        `BatchEnv` (TextWorld-gym wrapped) that implements `.reset()` /
        `.step()`. Some forks return the meta-env itself if it already
        looks gym-shaped — we only call `init_env(...)` when `.reset` is
        missing, so the adapter degrades gracefully.

        Also stashes the meta-env on the wrapper as `_alfred_meta` so
        `_select_task` can read the meta's `game_files` list (the wrapper
        typically doesn't expose it). Sets `self._is_batched = True` to
        flag downstream `_normalize_*` helpers to unwrap the per-batch
        list shape that TextWorld's batch env emits.

        Phase 2 EnvInfos registration (when
        `self._use_tw_intermediate_reward` is True): defensive 3-tier
        fallback because forks differ on whether `init_env` accepts a
        `request_infos` kwarg.
          - Tier 1: `init_env(batch_size=1, request_infos=env_infos)`.
          - Tier 2: set `meta.request_infos = env_infos`, then bare
            `init_env(batch_size=1)`.
          - Tier 3: bare `init_env(batch_size=1)` + `warnings.warn` that
            the upstream `intermediate_reward` field will likely not
            be populated. `_tw_registration_succeeded` stays False so
            `step()` falls back to the Phase 1 synthesized delta.
        Mirrors the existing `try/except TypeError` defensive layout in
        `_build_alfworld_env` for stylistic consistency.
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
                # Tier 2: attribute set, then bare init.
                # Wrap setattr and init_env separately so that an
                # init-time failure (e.g. TypeError/RuntimeError raised
                # by an env that doesn't accept the now-set
                # `request_infos` attribute) still drops into Tier 3
                # with the loud warning, matching the documented 3-tier
                # defensive contract. Only AttributeError on setattr is
                # the "Tier-2-not-supported" signal; any other failure
                # in init must escalate to Tier 3.
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
                        # init_env failed even after setattr succeeded —
                        # fall through to Tier 3 silent-fallback rather
                        # than propagating the exception.
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
            # Opt-in disabled (or `textworld` unimportable + opt-in off):
            # preserve Phase 1 behavior byte-for-byte.
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

        TextWorld batch env returns `(obs_list, info_dict_with_list_values)`.
        We unwrap for batch_size=1 so downstream code sees scalar obs +
        flat info dict. Only fires when `self._is_batched` (set in
        `_wrap_batch_env`) — for non-batched test fakes the raw shape
        passes through unchanged.
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

        TextWorld's batch env (returned by `AlfredTWEnv.init_env(batch_size=1)`)
        emits per-field LISTS of length `batch_size` (e.g. `obs=[str]`,
        `reward=[float]`, `done=[bool]`, `info={key: [val_per_batch]}`).
        We unwrap for batch_size=1 so downstream code sees scalar
        reward/done and a flat info dict. Only fires when
        `self._is_batched`.
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

        - `obs` = `[str]` → `str`
        - `reward` = `[float]` → `float`
        - `done` = `[bool]` → `bool`
        - `info` = `{k: [v_per_batch]}` → `{k: v_per_batch}`. For each
          info key, only unwrap when the value is a length-1 list/tuple
          AND the contained element is itself a list/tuple/dict (the
          TextWorld batch shape: outer batch list of inner per-batch
          structures). Plain scalar info values pass through unchanged.
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

        # Prime the expert-plan-length tracker for the dense progress
        # signal computed in step(). When the env doesn't expose an
        # expert_plan (non-handcoded experts, off-plan envs), the helper
        # returns [] → tracker stays None → step() yields delta=0.
        initial_plan = _extract_expert_plan(raw_info)
        self._prev_plan_len = len(initial_plan) if initial_plan else None

        # Phase-3: prime the per-step PDDL fact set. When opt-in is off
        # OR no facts are exposed, leave at None so step() yields
        # delta=0 instead of spuriously rewarding "everything new" on
        # the first step.
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

        # Dense progress shaping signal. Two sources, reconciled here:
        #
        #   1. (Phase 2, preferred when opt-in + registration succeeded)
        #      TextWorld's native `info["intermediate_reward"]` — a count
        #      of newly-satisfied PDDL fluents minus reverted ones per
        #      step. Strictly denser than plan-length deltas for tasks
        #      where one action flips multiple fluents (e.g. multi-object
        #      rearrangement).
        #   2. (Phase 1, fallback) The per-turn reduction in the AlfWorld
        #      handcoded expert's remaining-plan length. Captures
        #      "did this action move strictly closer to the goal in
        #      optimal action-step count?"
        #
        # Used by the standalone TurnRDv2 trainer as the V-head
        # supervision target (replaces the near-degenerate `raw_env_reward`
        # signal that's 0 everywhere except the final turn). Lives on
        # `info["intermediate_reward"]` so it never overwrites
        # `raw_env_reward` (Method-C `progress_decomposer` depends on the
        # literal env reward). The reconciled source is recorded under
        # `info["intermediate_reward_source"]` for post-hoc smoke debugging.
        #
        # Always recompute and update `_prev_plan_len` regardless of
        # which source fires so the Phase 1 fallback stays warm if
        # upstream goes None mid-trajectory.
        #
        # Edge cases (Phase 1 fallback path):
        #   • Plan is missing/empty mid-trajectory (off-plan, terminal
        #     state, env doesn't expose handcoded expert) → curr_plan_len
        #     becomes 0 and we conservatively emit 0. The max(0, ·)
        #     clamp also prevents shaping from going negative when
        #     AlfWorld re-plans after an off-plan action transiently
        #     lengthens the plan.
        #   • _prev_plan_len was never primed (reset() saw no expert
        #     plan) → fall back to curr_plan_len so delta=0 instead of
        #     spuriously rewarding the first step.
        # Capture upstream BEFORE we overwrite the key with the fallback.
        upstream_ir: Any = info.get("intermediate_reward") if (
            self._use_tw_intermediate_reward and self._tw_registration_succeeded
        ) else None
        # In the batched code path (`_is_batched=True`) TextWorld's
        # batch env emits scalar info values wrapped as length-1 lists
        # per batch slot (e.g. `[3]` for batch_size=1). `_unbatch_info`
        # intentionally only unwraps when the inner element is a
        # list/tuple/dict — scalar batch wrappings pass through
        # unchanged so generic non-batched info dicts aren't mangled.
        # Peel the per-batch wrapping here so the type-check below sees
        # the intended scalar.
        if isinstance(upstream_ir, (list, tuple)) and len(upstream_ir) == 1:
            upstream_ir = upstream_ir[0]
        curr_plan = _extract_expert_plan(info)
        curr_plan_len = len(curr_plan)
        prev_plan_len = (
            self._prev_plan_len if self._prev_plan_len is not None else curr_plan_len
        )
        fallback_delta = float(max(0, prev_plan_len - curr_plan_len))

        # Phase-3: PDDL-facts diff. Compute net delta = max(0, |new| -
        # |removed|) so an action that satisfies one fluent while
        # reverting another scores 0, not 1. The max(0, ·) clamp on
        # the net diff matches Phase 1's contract (V-head expects
        # non-negative shaping). Only fires when (a) opt-in is on,
        # (b) prev fact set was primed, and (c) curr fact set is
        # non-empty — first-step / no-facts-env collapses to None.
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
            # max(0, ·) clamp matches Phase 1's contract (downstream
            # V-head expects non-negative). TextWorld's intermediate_reward
            # can be negative when fluents get reverted; allowing
            # negative shaping is a separate experiment — out of scope.
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
