"""OpenAI GPT-4o-mini judge backend.

Default backend per proposal §3.2. Costs ~$0.65/1k judge calls with caching;
expected total study spend ~$3-6.

Requires `OPENAI_API_KEY` env var (loaded from a Modal Secret in production).
"""

from __future__ import annotations

from typing import Any

from src.judge.backend import JudgeRequest, TurnScore


class OpenAIJudge:
    model_tag: str

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.model = str(cfg.get("model", "gpt-4o-mini"))
        self.model_tag = self.model
        self.max_retries = int(cfg.get("max_retries", 3))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.timeout_s = float(cfg.get("timeout_s", 30.0))
        self.max_concurrency = int(cfg.get("max_concurrency", 8))
        self._client = None  # type: ignore[assignment]
        # TODO: lazy-init OpenAI client to avoid hard import dependency at scaffold time.

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        # TODO: from openai import OpenAI; self._client = OpenAI()
        # Deferred until requirements.txt grows the openai dep in Week 2.
        raise NotImplementedError(
            "OpenAIJudge.score_turns is a stub. Wire `openai>=1.40` in requirements.txt "
            "and implement client init + chat.completions.create call."
        )

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        self._ensure_client()
        # TODO Day 10:
        #   1. Render prompt via src.judge.prompts.render_prompt(request).
        #   2. Call chat.completions.create with response_format={"type": "json_object"}.
        #   3. Parse per-turn raw scores; assert len matches len(request.turns).
        #   4. Apply normalize_scores(raw, request.final_reward).
        #   5. Return list[TurnScore].
        raise NotImplementedError("OpenAIJudge.score_turns: implement on Day 10.")

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        # TODO: use openai.AsyncOpenAI; gate concurrency with asyncio.Semaphore(max_concurrency).
        raise NotImplementedError("OpenAIJudge.score_turns_async: implement on Day 11.")
