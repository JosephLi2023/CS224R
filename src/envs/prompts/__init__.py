"""Per-environment prompt templates used by the rollout collector."""

from src.envs.prompts.react_webshop import (
    WEBSHOP_SYSTEM_PROMPT,
    render_webshop_turn_prompt,
)

__all__ = [
    "WEBSHOP_SYSTEM_PROMPT",
    "render_webshop_turn_prompt",
]
