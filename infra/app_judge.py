"""Modal vLLM judge app (A10G). Skeleton — deploy on Day 11.

Will serve Qwen2.5-7B-Instruct via vLLM behind a `/score_turns` HTTP endpoint
consumed by `src.judge.vllm_backend.VLLMJudge`.

CLI (when ready):
  modal deploy infra/app_judge.py
  curl -X POST https://<workspace>--cs224r-hgpo-judge.modal.run/score_turns ...
"""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

APP_NAME = "cs224r-hgpo-judge"
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=300,
)
def hello() -> str:
    """Smoke test before we attach a GPU."""
    return "judge app is reachable"


# TODO Day 11: switch hello → @app.cls with gpu="A10G", concurrency_limit=8,
# load qwen2.5-7b-instruct via vllm.LLM at startup, expose /score_turns via FastAPI.


@app.local_entrypoint()
def main() -> None:
    print(hello.remote())
