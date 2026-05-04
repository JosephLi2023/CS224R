"""OpenAI GPT-4o-mini judge backend.

Default backend per proposal §3.2. Costs ~$0.65/1k judge calls with caching;
expected total study spend ~$3-6.

Requires `OPENAI_API_KEY` env var (loaded from a Modal Secret in production).

Implementation notes:
- Both sync and async clients are constructed lazily (`_ensure_clients`) so
  importing this module never requires an API key.
- The sync `score_turns` and async `score_turns_async` paths share a single
  parse routine (`_parse_response_to_turn_scores`) and a single retry loop
  shape: catch RateLimitError / APITimeoutError / APIConnectionError, sleep
  with exponential backoff, retry up to `max_retries` times, then re-raise.
- Concurrency control (asyncio.Semaphore) is owned by the caller (the
  JudgeDecomposer) so a single semaphore can gate K-trajectory groups across
  many score_turns_async calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.judge.backend import JudgeRequest, TurnScore
from src.judge.prompts import render_user_prompt, system_prompt, to_turn_scores


class OpenAIJudge:
    model_tag: str

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = str(cfg.get("model", "gpt-4o-mini"))
        self.model_tag = self.model
        self.max_retries = int(cfg.get("max_retries", 3))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout_s = float(cfg.get("timeout_s", 30.0))
        self.max_concurrency = int(cfg.get("max_concurrency", 8))
        # Initial backoff in seconds; doubled per retry.
        self.backoff_base_s = float(cfg.get("backoff_base_s", 1.0))
        self._client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------------
    # Client init
    # ------------------------------------------------------------------

    def _ensure_clients(self) -> None:
        """Lazy-import + lazy-construct both OpenAI clients.

        Reads `OPENAI_API_KEY` implicitly from the process env (Modal Secret
        injects it; tests can monkeypatch this method to avoid network).

        `max_retries=0` disables the OpenAI SDK's built-in retry layer so
        the in-house retry loop in `score_turns` / `score_turns_async` is
        the single source of truth. Without this, transient bursts could
        trigger up to (sdk_retries + 1) * (self.max_retries + 1) HTTP
        attempts per call, blowing through `max_judge_calls_per_run`
        budgets and rate-limit headroom (review item I1).
        """
        if self._client is not None and self._async_client is not None:
            return
        from openai import AsyncOpenAI, OpenAI  # local import to avoid hard dep at import time

        if self._client is None:
            self._client = OpenAI(timeout=self.timeout_s, max_retries=0)
        if self._async_client is None:
            self._async_client = AsyncOpenAI(timeout=self.timeout_s, max_retries=0)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_messages(self, request: JudgeRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": render_user_prompt(request)},
        ]

    def _parse_response_to_turn_scores(
        self, content: str, request: JudgeRequest
    ) -> list[TurnScore]:
        """Parse the JSON object the model returned and apply the §3.2 invariant.

        Raises ValueError on any structural issue (per Protocol contract:
        backends must raise on parse failure rather than silently zero).
        """
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAIJudge: response is not valid JSON: {content!r}") from e

        scores_field = payload.get("scores")
        if not isinstance(scores_field, list):
            raise ValueError(
                f"OpenAIJudge: response missing 'scores' list field: {payload!r}"
            )
        n_expected = len(request.turns)
        if len(scores_field) != n_expected:
            raise ValueError(
                f"OpenAIJudge: expected {n_expected} score entries, got "
                f"{len(scores_field)}: {payload!r}"
            )

        # Sort by 'turn' if present so out-of-order responses still align with
        # request.turns. After sorting, indices must be exactly [0..n-1].
        try:
            sorted_entries = sorted(scores_field, key=lambda e: int(e["turn"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"OpenAIJudge: each score entry must have an integer 'turn' key: "
                f"{payload!r}"
            ) from e

        observed_indices = [int(e["turn"]) for e in sorted_entries]
        if observed_indices != list(range(n_expected)):
            raise ValueError(
                f"OpenAIJudge: turn indices must be 0..{n_expected - 1}; "
                f"got {observed_indices}"
            )

        try:
            raw_scores = [float(e["score"]) for e in sorted_entries]
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"OpenAIJudge: each score entry must have a numeric 'score' key: "
                f"{payload!r}"
            ) from e

        return to_turn_scores(raw_scores, request.final_reward)

    def _is_transient(self, exc: BaseException) -> bool:
        """True if `exc` is a transient OpenAI API error worth retrying."""
        # Import lazily so the class is constructable without the openai dep
        # being installed at module import time.
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
            )
        except Exception:
            return False
        return isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError))

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff: base * 2**attempt."""
        return self.backoff_base_s * (2 ** attempt)

    # ------------------------------------------------------------------
    # Public sync + async API
    # ------------------------------------------------------------------

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        self._ensure_clients()
        messages = self._build_messages(request)
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAIJudge: response content was None")
                return self._parse_response_to_turn_scores(content, request)
            except Exception as exc:  # noqa: BLE001 — transient classification done in helper
                # Narrowed from BaseException so KeyboardInterrupt /
                # SystemExit / asyncio.CancelledError propagate as Python
                # intends (review item I2).
                if self._is_transient(exc) and attempt < self.max_retries:
                    last_exc = exc
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                raise
        # Defensive: we should never fall through, but keep mypy/pyright happy.
        raise RuntimeError(
            f"OpenAIJudge: exhausted retries without raising; last_exc={last_exc!r}"
        )

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        self._ensure_clients()
        messages = self._build_messages(request)
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAIJudge: response content was None")
                return self._parse_response_to_turn_scores(content, request)
            except Exception as exc:  # noqa: BLE001 — transient classification done in helper
                # Narrowed from BaseException so asyncio.CancelledError
                # (a BaseException subclass) propagates the gather cancel
                # cascade transparently per Python's contract (review I2).
                if self._is_transient(exc) and attempt < self.max_retries:
                    last_exc = exc
                    await asyncio.sleep(self._backoff_seconds(attempt))
                    continue
                raise
        raise RuntimeError(
            f"OpenAIJudge: exhausted retries without raising; last_exc={last_exc!r}"
        )
