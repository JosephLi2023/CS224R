"""OpenAI GPT-4o-mini judge backend (default).

Clients are constructed lazily so importing never needs an API key. The sync
and async paths share one parse routine and one retry loop (exponential
backoff on transient API errors). Requires `OPENAI_API_KEY`.
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

    # Client init

    def _ensure_clients(self) -> None:
        """Lazy-import + construct both OpenAI clients.

        `max_retries=0` disables the SDK's retry layer so our own retry loop
        is the single source of truth.
        """
        if self._client is not None and self._async_client is not None:
            return
        from openai import AsyncOpenAI, OpenAI  # local import to avoid hard dep at import time

        if self._client is None:
            self._client = OpenAI(timeout=self.timeout_s, max_retries=0)
        if self._async_client is None:
            self._async_client = AsyncOpenAI(timeout=self.timeout_s, max_retries=0)

    # Shared helpers

    def _build_messages(self, request: JudgeRequest) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": render_user_prompt(request)},
        ]

    def _parse_response_to_turn_scores(
        self, content: str, request: JudgeRequest
    ) -> list[TurnScore]:
        """Parse the model's JSON response and normalize to the sum invariant.

        Raises ValueError on any structural issue (never silently zeros).
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

        # Sort by 'turn' so out-of-order responses align; indices must be 0..n-1.
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
        # Import lazily so the class is constructable without the openai dep.
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

    # Public sync + async API

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
            except Exception as exc:  # noqa: BLE001 - transient classification done in helper
                # Narrowed from BaseException so KeyboardInterrupt/SystemExit propagate.
                if self._is_transient(exc) and attempt < self.max_retries:
                    last_exc = exc
                    time.sleep(self._backoff_seconds(attempt))
                    continue
                raise
        # Defensive: should never fall through.
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
            except Exception as exc:  # noqa: BLE001 - transient classification done in helper
                # Narrowed from BaseException so asyncio.CancelledError propagates.
                if self._is_transient(exc) and attempt < self.max_retries:
                    last_exc = exc
                    await asyncio.sleep(self._backoff_seconds(attempt))
                    continue
                raise
        raise RuntimeError(
            f"OpenAIJudge: exhausted retries without raising; last_exc={last_exc!r}"
        )
