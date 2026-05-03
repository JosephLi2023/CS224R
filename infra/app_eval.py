"""Modal eval-only app: runs evaluator against a checkpoint on the Volume."""

from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import image

APP_NAME = "cs224r-hgpo-eval"
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60,
)
def evaluate(checkpoint_path: str, episodes: int = 100) -> str:
    """Eval-only entrypoint. Stub until LoRAPolicy + LLM evaluator land Week 1."""
    # TODO Week 1 Day 7: load LoRAPolicy from checkpoint_path, run evaluator
    # against the eval split, write results to /vol/manifests/<run>/eval_log.json.
    raise NotImplementedError("Eval entrypoint pending Week 1 Day 7.")


@app.local_entrypoint()
def main(checkpoint: str = "", episodes: int = 100) -> None:
    if not checkpoint:
        raise SystemExit("Provide --checkpoint /vol/checkpoints/...")
    print(evaluate.remote(checkpoint, episodes))
