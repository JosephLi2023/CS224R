"""Modal training app: trainer entrypoint + a hello-world smoke test.

CLI:
  modal run infra/app_train.py::hello                 # CPU smoke; verifies image builds + Volume mounts
  modal run infra/app_train.py::env_probe             # A100 smoke; verifies torch+CUDA inside image
  modal run infra/app_train.py::train --config configs/method_flat_grpo.json --seed 11

The `train` function dispatches into src.trainers.train (current toy entrypoint)
until the LLM trainer lands in Week 1; once `src/trainers/train_hgpo.py` exists
it will be imported here instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, maybe_openai_secret, volume
from infra.image import image

APP_NAME = "cs224r-hgpo-train"
app = modal.App(APP_NAME)

# CPU image variant for cheap smoke tests
SMOKE_TIMEOUT = 300  # 5 min
TRAIN_TIMEOUT = 6 * 60 * 60  # 6 hr ceiling per run


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=SMOKE_TIMEOUT,
)
def hello() -> str:
    """CPU-only smoke test. Verifies the image builds, source is mounted,
    and the Volume is writable. Exits in <1 minute. Costs nothing."""
    import datetime
    import os

    msg = (
        f"hello from modal at {datetime.datetime.now(datetime.UTC).isoformat()}\n"
        f"workspace contents: {os.listdir('/workspace')}\n"
    )
    sentinel = Path(VOLUME_MOUNT) / "hello.sentinel"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(msg)
    volume.commit()
    print(msg)
    return msg


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    secrets=maybe_openai_secret(),
    timeout=SMOKE_TIMEOUT,
)
def env_probe() -> dict:
    """A100 smoke test. Verifies torch sees CUDA and reports versions.
    Costs ~$0.05 (1 minute on A100)."""
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = list(torch.cuda.get_device_capability(0))  # tuple → list, deserializable on non-torch host
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info["torch_error"] = repr(e)

    for mod in ("transformers", "peft", "trl", "vllm", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[f"{mod}_error"] = repr(e)

    # Confirm the OpenAI Modal Secret reached the container env when
    # `openai-secret` is provisioned. Used by Method A end-to-end smoke.
    import os
    info["openai_api_key_present"] = bool(os.getenv("OPENAI_API_KEY"))
    print(info)
    return info


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: volume},
    secrets=maybe_openai_secret(),
    timeout=TRAIN_TIMEOUT,
)
def train(
    train_config: str = "configs/method_flat_grpo.json",
    env_config: str = "configs/env_webshop.json",
    eval_config: str = "configs/eval.json",
    seed: int = 11,
) -> str:
    """Run a single training job inside Modal A100.

    For now this delegates to the existing toy `src.trainers.train`. Swap
    the import to `src/trainers/train_hgpo.py` once that module is ready.
    """
    sys.path.insert(0, "/workspace")
    # Confirm the OpenAI secret reached us when judge.backend=openai.
    # Cheap one-line probe so failures surface in `modal app logs` rather than
    # at first JudgeBackend.score_turns call.
    import os
    print(
        f"[train] OPENAI_API_KEY present in container env: "
        f"{bool(os.getenv('OPENAI_API_KEY'))}"
    )
    # Import lazily so the CPU `hello` smoke test doesn't pull torch.
    import argparse  # noqa: F401
    from src.trainers import train as toy_train  # type: ignore

    sys.argv = [
        "train",
        "--env-config", f"/workspace/{env_config}",
        "--train-config", f"/workspace/{train_config}",
        "--eval-config", f"/workspace/{eval_config}",
    ]
    toy_train.main()
    volume.commit()
    return f"train complete: {train_config} seed={seed}"


@app.local_entrypoint()
def main(action: str = "hello") -> None:
    """Convenience entrypoint: `modal run infra/app_train.py --action hello`."""
    if action == "hello":
        print(hello.remote())
    elif action == "env_probe":
        print(env_probe.remote())
    else:
        raise ValueError(f"Unknown action: {action}")
