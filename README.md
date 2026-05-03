# CS224R HGPO Project Scaffold

This repository is a milestone-first scaffold for baseline and HGPO experiments.

## Structure

- `configs/`: environment, training, and evaluation configs.
- `src/envs/`: environment wrappers/adapters.
- `src/algorithms/baseline/`: baseline policy optimizer logic.
- `src/algorithms/hgpo/`: HGPO grouping/objective integration.
- `src/trainers/`: train/eval entrypoints and shared utilities.
- `scripts/`: launch scripts for local or Modal-wrapped runs.
- `experiments/manifests/`: run outputs, checkpoints, and config snapshots.
- `reports/milestone/`, `reports/poster/`, `reports/final/`: deliverable workspaces.

## Setup

```bash
cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Training

Current smoke path (toy env):

```bash
cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
bash scripts/run_modal_train.sh configs/baseline_train.json
```

HGPO smoke path:

```bash
cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
bash scripts/run_modal_train.sh configs/hgpo_train.json
```

Target environments configured:
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/configs/env_webshop.json`
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/configs/env_alfworld.json`

WebShop adapter details:
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/webshop_adapter.py` now supports real WebShop API wiring with:
  - import path `webshop.envs.web_agent_site_env.WebAgentTextEnv`,
  - reset normalization (`obs` vs `(obs, info)`),
  - step normalization to `(state, reward, done, info)`,
  - action resolution from `str` command or `int` candidate index.

ALFWorld adapter details:
- `/Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R/src/envs/alfworld_adapter.py` now supports ALFWorld API wiring with:
  - common module bootstrap paths in `alfworld.agents.environment`,
  - reset normalization across variants,
  - step normalization for 4-tuple and 5-tuple APIs,
  - admissible action extraction and `str`/`int` action resolution.

Each run writes to `experiments/manifests/<run_name>_<timestamp>/`:
- `config_snapshot.json`
- `train_log.json`
- `eval_log.json`
- `checkpoints/episode_*.json`

## Run Evaluation

```bash
cd /Users/shoupeili/Desktop/classifiers/CS224R/project/CS224R
bash scripts/run_modal_eval.sh configs/hgpo_train.json \
  experiments/manifests/<run_dir>/checkpoints/episode_300.json
```

## Reproducibility Controls

- Seed is tracked in config and checkpoint payloads.
- Full merged configuration snapshot is written per run.
- Resume training supported with `--resume-checkpoint` on `src.trainers.train`.

## Next Implementation Steps

1. Wire `src/envs/webshop_adapter.py` to the real WebShop API and `src/envs/alfworld_adapter.py` to real ALFWorld APIs.
2. Extend trainer rollout logic from toy action sampling to text-action trajectories for WebShop/ALFWorld.
3. Swap current HGPO shaping stub with full imported HGPO objective pieces from reference recipe.
4. Add unit and smoke tests in `tests/unit/` and `tests/smoke/`.
5. Wrap scripts with actual Modal job definitions.
