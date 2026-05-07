"""Pluggable LLM-as-judge backends for H-GRPO Method A.

Exposes a `JudgeBackend` Protocol plus two implementations:
- `OpenAIJudge` (default): GPT-4o-mini via OpenAI API.
- `VLLMJudge` (fallback): Qwen2.5-7B-Instruct served behind a Modal vLLM app.

Select the backend via config `judge.backend ∈ {openai, vllm}`.
"""

from src.judge.backend import JudgeBackend, TurnScore, build_judge
from src.judge.cache import JudgeCache

__all__ = [
    "JudgeBackend",
    "TurnScore",
    "build_judge",
    "JudgeCache",
]
