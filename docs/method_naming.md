# Method naming — canonical vocabulary

The codebase accumulated a few generations of method naming
(`method_a/b/c/d`, `Method A/B/C/D`, `flat_grpo`, `judge`,
`progress`, `turnrd`, `turnrd_v2`, `counterfactual`, etc.). The
writeup uses a single canonical vocabulary; this file is the
old↔new mapping for navigating between code/configs/manifests
and the report text.

## Mapping table

| Canonical name (writeup) | Description | Legacy keys (configs / manifest rows / register keys) |
|---|---|---|
| **SFTOnly** | SFT-warm-started Qwen2.5-1.5B, no RL — the floor baseline. | (no prior key — new in this rename pass) |
| **flatGRPO** | H-GRPO with α=1.0 (drops per-turn signal) — the trivial-decomposer baseline. | `flat_grpo` (manifest), `configs/method_flat_grpo.json`, `flat_grpo_compare_seed11_*` (run names) |
| **TurnRDV1** | Original learned attention decomposer (best v1 variant). | `method_b_lean` (manifest, run name), `configs/method_hgpo_turnrd_lean.json`, also `method_b`, `method_b_best`, `Method B` |
| **TurnRDV2** | Bidirectional + Σα·v identifiable + progress-prior init + policy-carry. | `method_b_v2_carry` (manifest, run name), `configs/method_hgpo_turnrd_v2_carry.json`, also `method_b_v2`, `method_b_v2_1x200` |
| **Progressive** | Progress-decomposer (env raw_env_reward × α=0.5). | `method_c` (manifest), `configs/method_hgpo_progress.json`, also `Method C` |
| **LLMJudge** | OpenAI judge-decomposer (gpt-4o-mini per-turn scoring). | (not in legacy manifest), `configs/method_hgpo_judge.json`, also `Method A`, `judge` (decomposer registry key) |
| **Counterfactual** | Replay-based CF rollouts decomposer (N=2 alts, 3-turn completions). | (not in legacy 4-method manifest), `configs/method_hgpo_counterfactual.json`, also `Method D`, `cf_compare_seed11_*` (run names), `counterfactual` (registry key) |

## What stays legacy-named

The rename is **writeup-facing + new-config + new-run-name** only.
The following deliberately keep their old names so prior runs,
artifacts, plots, and tests don't break:

- `src/turnrd/` directory and the `TurnRD` / `TurnRDv2` Python classes
- `src/algorithms/hgpo/decomposers/` registry keys
  (`"turnrd"`, `"judge"`, `"progress"`, `"counterfactual"`)
- All existing config files with old names (kept for reproducibility)
- Saved Modal volume paths (e.g.
  `/vol/cache/method_b_v2_carry/{replay,ckpt}`)
- All existing `train_log.json` / `summary.json` artifacts on
  `/vol/manifests/method_b_*_seed11_round0N_*` and
  `experiments/manifests/_baseline_turnrd/` /
  `experiments/manifests/_flat_grpo/`
- All existing test files (`tests/unit/test_turnrd_v2_model.py`,
  `tests/unit/test_turnrd_decomposer.py`,
  `tests/unit/test_train_hgpo_config_loader.py`, etc.)

The legacy 4-method comparison artifacts (the JSON, the TXT, and
the milestone PNGs) live under
`experiments/manifests/_legacy/` for archaeology.

## Where the canonical vocabulary lives

- `experiments/manifests/methods_comparison.json` — one row per
  canonical method, BEST variant of each family.
- `configs/{SFTOnly,flatGRPO,TurnRDV1,TurnRDV2,Progressive,LLMJudge,Counterfactual}.json`
  — clean-named configs for FUTURE runs. Each clones the
  corresponding legacy config; TurnRDV1 and TurnRDV2 use
  `/vol/cache/TurnRD{V1,V2}/` so canonical-name runs do not
  pollute legacy caches.
- `scripts/run_methods_protocol.sh` — canonical-name launcher
  that dispatches each method to the correct Modal entrypoint.
  The legacy `scripts/run_webshop_protocol.sh` still works for
  reproducing prior runs.
