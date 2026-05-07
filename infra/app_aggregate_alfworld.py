"""Modal app: aggregate per-round AlfWorld sweep results into the manifest JSON.

Reads `train_log.json` files written by `infra/app_train_loop.py::_train_loop_impl`
to `/vol/manifests/<run_name_prefix>_round??_<ts>/train_log.json`, extracts the
`eval` block from each round, and computes per-method best/last/mean stats in
the same row schema as `experiments/manifests/4method_comparison.json`.

Usage:
    modal run infra/app_aggregate_alfworld.py

Outputs the assembled `4method_comparison_alfworld.json` payload to stdout
(also persists to `/vol/manifests/4method_comparison_alfworld.json` on the
volume so subsequent runs can re-fetch it).
"""
from __future__ import annotations

import modal  # type: ignore[import-not-found]

from infra.common import VOLUME_MOUNT, volume
from infra.image import alfworld_image

app = modal.App("cs224r-hgpo-aggregate-alfworld")


# Method → run-name prefix (matches what scripts/run_turnrd_modal.py and
# scripts/run_method_c_alfworld.sh wrote on the volume).
METHOD_PREFIXES: dict[str, str] = {
    "method_b_v2_alfworld": "method_b_v2_alfworld_seed11",
    "method_b_lean_alfworld": "method_b_lean_alfworld_seed11",
    "method_c_alfworld": "method_c_alfworld_seed11",
}


@app.function(image=alfworld_image, volumes={VOLUME_MOUNT: volume}, timeout=10 * 60)
def aggregate(notes_by_method: dict[str, str] | None = None) -> dict:
    """Aggregate per-round eval blocks into the comparison manifest dict.

    Mirrors the row shape used by `experiments/manifests/4method_comparison.json`:
    `{n_rounds, best_eval_return, last_eval_return, mean_eval_return,
      best_pct_success, last_pct_success, mean_pct_success,
      total_eval_episodes, _per_round_eval: [{round, avg_R, pct_success, n_ok}, ...]}`.

    Rounds that crashed (no `eval` block in `train_log.json`) are skipped,
    matching the WebShop manifest's behavior (n_rounds reflects only
    rounds that produced a valid eval).
    """
    import json
    import os
    import re
    import sys

    notes_by_method = notes_by_method or {}
    volume.reload()

    manifests_root = "/vol/manifests"
    if not os.path.isdir(manifests_root):
        raise RuntimeError(f"manifests root missing: {manifests_root}")

    round_dir_re = re.compile(r"^(?P<prefix>.+)_round(?P<idx>\d{2})_\d+_\d+$")

    out: dict[str, dict] = {}

    for method_name, prefix in METHOD_PREFIXES.items():
        per_round: list[dict] = []
        for child in sorted(os.listdir(manifests_root)):
            child_path = os.path.join(manifests_root, child)
            if not os.path.isdir(child_path):
                continue
            m = round_dir_re.match(child)
            if not m:
                continue
            if m.group("prefix") != prefix:
                continue
            round_idx = int(m.group("idx"))
            log_path = os.path.join(child_path, "train_log.json")
            if not os.path.isfile(log_path):
                print(
                    f"  [{method_name}] round {round_idx}: no train_log.json at {log_path}; skipping",
                    file=sys.stderr,
                )
                continue
            try:
                with open(log_path) as fh:
                    log = json.load(fh)
            except Exception as exc:
                print(
                    f"  [{method_name}] round {round_idx}: failed to parse {log_path}: {exc!r}",
                    file=sys.stderr,
                )
                continue
            eval_block = log.get("eval")
            if not isinstance(eval_block, dict):
                print(
                    f"  [{method_name}] round {round_idx}: no `eval` block in {log_path}; "
                    "the round's train_loop likely crashed before eval. Skipping.",
                    file=sys.stderr,
                )
                continue
            per_round.append(
                {
                    "round": round_idx,
                    "avg_R": float(eval_block.get("avg_return", 0.0)),
                    "pct_success": float(eval_block.get("pct_success", 0.0)),
                    "n_ok": int(eval_block.get("n_episodes_ok", 0)),
                    "n_attempted": int(eval_block.get("n_episodes_attempted", 0)),
                    "run_dir": child_path,
                }
            )

        per_round.sort(key=lambda r: r["round"])

        if not per_round:
            print(f"  [{method_name}] no valid rounds; skipping method", file=sys.stderr)
            continue

        avg_R_series = [r["avg_R"] for r in per_round]
        pct_series = [r["pct_success"] for r in per_round]
        total_eval_eps = sum(r["n_ok"] for r in per_round)

        # Pick best by avg_R (tie-broken by pct_success), last by round.
        best_idx = max(
            range(len(per_round)),
            key=lambda i: (avg_R_series[i], pct_series[i]),
        )

        out[method_name] = {
            "n_rounds": len(per_round),
            "best_eval_return": round(avg_R_series[best_idx], 4),
            "last_eval_return": round(avg_R_series[-1], 4),
            "mean_eval_return": round(sum(avg_R_series) / len(avg_R_series), 5),
            "best_pct_success": round(pct_series[best_idx], 4),
            "last_pct_success": round(pct_series[-1], 4),
            "mean_pct_success": round(sum(pct_series) / len(pct_series), 4),
            "total_eval_episodes": total_eval_eps,
            "_per_round_eval": [
                {"round": r["round"], "avg_R": round(r["avg_R"], 4),
                 "pct_success": round(r["pct_success"], 4), "n_ok": r["n_ok"]}
                for r in per_round
            ],
        }
        notes = notes_by_method.get(method_name)
        if notes:
            out[method_name]["_notes"] = notes

    # Persist to the volume so the host can `modal volume get` it.
    out_path = "/vol/manifests/4method_comparison_alfworld.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    volume.commit()
    print(f">>> wrote {out_path}", file=sys.stderr)

    return out


@app.local_entrypoint()
def main() -> None:
    import json as _json

    notes = {
        "method_b_v2_alfworld": (
            "Method-B v2 (TurnRDv2) on AlfWorld: bidirectional encoder + "
            "identifiable Σα·v R-loss + progress-prior init. 5×40 protocol on "
            "trimmed AlfWorld pool (200 train games, 50 valid_seen + 50 "
            "valid_unseen), seed 11, K=4. Re-match of WebShop's v2 vs C result "
            "on a multi-step credit-distribution env."
        ),
        "method_b_lean_alfworld": (
            "Method-B lean (TurnRD v1, CLS-bottlenecked causal encoder, no "
            "aux losses, refresh_every_episodes=10) on AlfWorld. 5×40, seed 11, "
            "K=4. Apples-to-apples baseline vs v2."
        ),
        "method_c_alfworld": (
            "Method C (progress decomposer = raw env reward per turn) on "
            "AlfWorld. 5×40 via direct modal calls (orchestrator skipped — "
            "no TurnRD plumbing needed). seed 11, K=4."
        ),
    }
    result = aggregate.remote(notes)
    print(_json.dumps(result, indent=2))
