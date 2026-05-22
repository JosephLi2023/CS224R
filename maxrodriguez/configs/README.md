# Max Rodriguez Configs

Put smoke and final experiment configs here once a turn-level reward method is
implemented.

Expected first files:

```text
alfworld_progress_smoke.json
turnrd_signed_smoke.json
counterfactual_delta_smoke.json
```

Each config should preserve the proposal blend:

```text
A_H = alpha * A_turn + (1 - alpha) * A_traj
```

and should set:

```json
{
  "hgpo": {
    "alpha": 0.5,
    "lambda_consistency": 0.0,
    "decomposer": "<method_name>"
  }
}
```
