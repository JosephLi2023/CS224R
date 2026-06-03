# CS224R: Per-turn credit assignment for multi-turn LLM agents

Course project studying how to assign per-turn credit when fine-tuning a
language-model agent on long-horizon, sparse-reward tasks with RL.

## Overview

The agent is a Qwen2.5 policy (PEFT LoRA) acting in two text environments,
WebShop and AlfWorld, where reward only arrives at the end of an episode.
We start from a supervised warm-start (SFT on deterministic-oracle
trajectories) and then run group-relative policy optimization (GRPO).

The core question is credit assignment: instead of applying the single
final-episode reward uniformly to every turn (flat GRPO), we decompose it
into per-turn rewards and compare several decomposers:

- TurnRD: a small learned model that attends over the turns and predicts
  a per-turn share of the episode return.
- Judge: an LLM-as-judge scores each turn.
- Progress: uses the dense per-step environment signal directly.
- Counterfactual: re-rolls each turn with alternative actions to estimate
  its marginal contribution.

All training and evaluation run on Modal against a shared volume; the local
side is just shell launchers and light Python.

## Repo layout

```
src/
  algorithms/grpo/      GRPO trainer, rollout buffer, advantage, KL
  algorithms/hgpo/      hierarchical GRPO + per-turn reward decomposers
  turnrd/               learned TurnRD decomposer model, dataset, training
  policy/               LoRA policy + vLLM runner + weight sync
  envs/                 WebShop / AlfWorld adapters + ReAct prompts
  datasets/             SFT trajectory loaders
  judge/                LLM-as-judge backends (OpenAI / vLLM)
  trainers/             standalone training/eval entry points
infra/                  Modal apps (env install, SFT gen/train, train loop)
scripts/                shell launchers for the SFT + RL pipelines
configs/                env + method JSON configs
tests/                  unit + integration tests
```

## Running

Pipelines are driven by the launchers in `scripts/` (env install, SFT, then
the RL methods) and run on Modal. See the header comment of each
`scripts/run_*.sh` for its flags and defaults.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/unit tests/integration -q
```
