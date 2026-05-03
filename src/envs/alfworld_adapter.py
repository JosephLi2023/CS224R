from __future__ import annotations


class ALFWorldAdapter:
    """
    Integration stub for the ALFWorld environment.

    Replace `reset`/`step` internals with actual ALFWorld environment API calls.
    Keep this adapter interface stable so trainers/algorithms remain unchanged.
    """

    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self._steps = 0

    def reset(self) -> str:
        self._steps = 0
        return "alfworld_initial_observation"

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        self._steps += 1
        done = self._steps >= self.max_steps
        reward = 0.0
        obs = f"alfworld_obs_after_{action}"
        info = {"env": "alfworld"}
        return obs, reward, done, info
