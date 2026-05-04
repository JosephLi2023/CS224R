"""SQLite-backed cache for judge scores.

Cache keys: (task_id, turn_idx, prefix_hash, judge_model_tag).
The model tag keeps OpenAI/Qwen entries cleanly partitioned so flipping
backends never returns stale, mismatched scores.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from src.judge.backend import JudgeRequest, JudgeTurn, TurnScore


def prefix_hash(env_name: str, turns: list[JudgeTurn], up_to_turn: int) -> str:
    """Deterministic hash over the trajectory prefix up to (and including) turn `up_to_turn`.

    Designed so that two trajectories sharing a common prefix could reuse
    cached scores for shared turns when the per-turn judge prompt is purely
    prefix-conditioned.

    NOTE (as of 2026-05-04): this prefix-sharing is NOT realized end-to-end
    today. The current `JudgeDecomposer` qualifies `task_id` with the
    K-sample index (`{task_id}#k{i}`) for a correctness reason: cached
    `normalized` values are pre-scaled against a specific `final_reward`,
    so cross-trajectory reuse with different `R`s would silently violate
    the §3.2 `Σ_t r̂_t = R` invariant. As a result, cache entries from
    different K-samples are never shared even when their prefixes match.
    To genuinely re-enable cross-K prefix sharing, cache `raw_score` only
    and re-normalize at read time using `request.final_reward`, then drop
    the `#k{i}` qualifier in
    `src/algorithms/hgpo/decomposers/judge.py::_build_request`.
    """
    payload = {
        "env": env_name,
        "prefix": [
            {"i": t.turn_idx, "obs": t.observation_text, "act": t.action_text}
            for t in turns[: up_to_turn + 1]
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS judge_scores (
    task_id        TEXT NOT NULL,
    turn_idx       INTEGER NOT NULL,
    prefix_hash    TEXT NOT NULL,
    model_tag      TEXT NOT NULL,
    raw_score      REAL NOT NULL,
    normalized     REAL NOT NULL,
    final_reward   REAL NOT NULL,
    inserted_at    REAL NOT NULL,
    PRIMARY KEY (task_id, turn_idx, prefix_hash, model_tag)
);
CREATE INDEX IF NOT EXISTS idx_model_tag ON judge_scores(model_tag);
"""


class JudgeCache:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, task_id: str, turn_idx: int, prefix_hash_: str, model_tag: str) -> TurnScore | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT raw_score, normalized FROM judge_scores "
                "WHERE task_id=? AND turn_idx=? AND prefix_hash=? AND model_tag=?",
                (task_id, turn_idx, prefix_hash_, model_tag),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TurnScore(turn_idx=turn_idx, raw_score=float(row[0]), normalized=float(row[1]))

    def put(
        self,
        *,
        task_id: str,
        prefix_hash_: str,
        model_tag: str,
        final_reward: float,
        score: TurnScore,
    ) -> None:
        import time

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO judge_scores "
                "(task_id, turn_idx, prefix_hash, model_tag, raw_score, normalized, final_reward, inserted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    score.turn_idx,
                    prefix_hash_,
                    model_tag,
                    float(score.raw_score),
                    float(score.normalized),
                    float(final_reward),
                    time.time(),
                ),
            )
            self._conn.commit()

    def get_or_miss(self, request: JudgeRequest, model_tag: str) -> tuple[list[TurnScore | None], list[str]]:
        """Return (per-turn-cached-or-None, per-turn-prefix-hashes).

        Caller fills the None entries by querying the backend, then writes them
        back with `put(...)`. This keeps the cache fully precomputable and
        hides hash construction from the trainer.
        """
        hashes = [prefix_hash(request.env_name, request.turns, t.turn_idx) for t in request.turns]
        cached: list[TurnScore | None] = []
        for t, h in zip(request.turns, hashes):
            cached.append(self.get(request.task_id, t.turn_idx, h, model_tag))
        return cached, hashes

    def close(self) -> None:
        with self._lock:
            self._conn.close()
