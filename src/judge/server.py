"""Modal vLLM judge server (optional, only deployed when judge.backend=vllm).

Exposes `/score_turns` accepting a JSON-serialized JudgeRequest and returning
{"scores": [<float>, ...]} with one entry per turn. Stub for Day 11.
"""

from __future__ import annotations

# TODO Day 11:
# 1. Define a Modal app with @app.cls(gpu="A10G", concurrency_limit=8, volumes={"/vol": vol}).
# 2. On startup: load Qwen2.5-7B-Instruct via vllm.LLM(model=..., dtype="bfloat16", max_model_len=8192).
# 3. Define a FastAPI endpoint POST /score_turns that:
#    a. Accepts {task_id, env_name, turns: [...], final_reward}.
#    b. Renders prompt via src.judge.prompts.render_user_prompt + system_prompt.
#    c. Calls vllm with sampling_params(temperature=0.0, max_tokens=512, response_format=...).
#    d. Parses JSON, returns {"scores": [<raw float>, ...]}.
# 4. Health endpoint GET /healthz.
# 5. Optional: HMAC auth via judge.api_token.

raise NotImplementedError(
    "src.judge.server is a stub. Implement the Modal vLLM judge app on Day 11."
)
