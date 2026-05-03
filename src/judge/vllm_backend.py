"""vLLM-served Qwen2.5-7B-Instruct judge backend (fallback).

Talks to a Modal app (`modal/app_judge.py`) that exposes `/score_turns`.
Used when no OpenAI API key is available or for fully self-contained reproducibility.
"""

from __future__ import annotations

from typing import Any

from src.judge.backend import JudgeRequest, TurnScore


class VLLMJudge:
    model_tag: str

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.endpoint = str(cfg["endpoint"])  # e.g. https://<workspace>--judge-app.modal.run/score_turns
        self.model = str(cfg.get("model", "qwen2.5-7b-instruct"))
        self.model_tag = self.model
        self.timeout_s = float(cfg.get("timeout_s", 60.0))
        self.max_concurrency = int(cfg.get("max_concurrency", 4))
        self.api_token = str(cfg.get("api_token", ""))  # optional shared secret

    def score_turns(self, request: JudgeRequest) -> list[TurnScore]:
        # TODO Day 11:
        #   1. POST request as JSON to self.endpoint (use httpx).
        #   2. Parse JSON {"scores": [...]}; len must match request.turns.
        #   3. Apply normalize_scores(raw, request.final_reward).
        #   4. Retry on 5xx with exponential backoff.
        raise NotImplementedError("VLLMJudge.score_turns: implement on Day 11.")

    async def score_turns_async(self, request: JudgeRequest) -> list[TurnScore]:
        # TODO: use httpx.AsyncClient; gate concurrency with asyncio.Semaphore.
        raise NotImplementedError("VLLMJudge.score_turns_async: implement on Day 11.")
