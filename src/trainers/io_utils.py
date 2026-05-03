from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump_json(path: str, obj: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_checkpoint(path: str, logits: list[float], episode: int, seed: int, algorithm: str) -> None:
    payload = {
        "logits": logits,
        "episode": episode,
        "seed": seed,
        "algorithm": algorithm,
    }
    dump_json(path, payload)


def load_checkpoint(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
