"""Unit tests for `src.turnrd.embedders.policy_hidden_state_embedder`.

Verification matrix:
1. `test_embedder_returns_correct_shape_per_trajectory`
2. `test_embedder_mean_pool_honors_padding_mask`
3. `test_embedder_returns_cpu_fp32_regardless_of_model_dtype`

Implementation note: the embedder only depends on a duck-typed
`policy.tokenizer(...)` and `policy.model(input_ids=..., attention_mask=...,
output_hidden_states=True)`. We supply lightweight stubs for both — no
real HF model + no network required — so these tests are fast and stable
on a CI/Mac host.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.algorithms.grpo.rollout import Trajectory, TurnRecord  # noqa: E402
from src.turnrd.embedders import policy_hidden_state_embedder  # noqa: E402


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Minimal HF-tokenizer-shaped callable.

    Maps each span to a deterministic id sequence based on character
    counts (so spans of different lengths produce different L). Pads to
    the longest sequence in the batch. Returns the dict-like object that
    HF tokenizers return (`{"input_ids": ..., "attention_mask": ...}`).
    """

    def __init__(self, max_len: int = 32) -> None:
        self.max_len = max_len
        self.pad_id = 0

    def __call__(
        self,
        spans: list[str],
        *,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, "torch.Tensor"]:
        cap = self.max_len if max_length is None else min(self.max_len, max_length)
        # 1 token per character (deterministic, easy to reason about).
        per_span = []
        for s in spans:
            n = max(1, min(len(s), cap))
            per_span.append([1 + (ord(c) % 100) for c in s[:n]])
        L = max(len(ids) for ids in per_span)
        input_ids = torch.zeros(len(spans), L, dtype=torch.long)
        attn = torch.zeros(len(spans), L, dtype=torch.long)
        for i, ids in enumerate(per_span):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


@dataclass
class _StubOutput:
    hidden_states: tuple


class _StubModel:
    """Minimal HF-model-shaped object exposing parameters() + forward.

    The "hidden state" for token id `k` at position `p` is a deterministic
    vector built from `k` and `p` so tests can verify mean-pooling math.
    """

    def __init__(self, hidden_size: int = 8, dtype=None) -> None:
        self.hidden_size = hidden_size
        self.training = False
        # Single parameter so `next(model.parameters())` works.
        self._param = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype if dtype is not None else torch.float32)
        )
        self._dtype = self._param.dtype

    # HF-model surface
    def parameters(self):
        yield self._param

    def __call__(
        self,
        input_ids: "torch.Tensor",
        attention_mask: "torch.Tensor" | None = None,
        output_hidden_states: bool = False,
        **_: Any,
    ) -> _StubOutput:
        T, L = input_ids.shape
        # Hidden state h[t, p, d] = (input_ids[t, p].float() + p * 0.01) * 0.1
        # plus a feature-axis ramp `d * 1.0` so different hidden dims are
        # distinguishable. Lives in the model's parameter dtype.
        ids = input_ids.to(dtype=self._dtype)
        positions = torch.arange(L, dtype=self._dtype).unsqueeze(0)  # [1, L]
        base = (ids + positions * 0.01) * 0.1  # [T, L]
        feat_ramp = torch.arange(self.hidden_size, dtype=self._dtype)  # [D]
        hidden = base.unsqueeze(-1) + feat_ramp  # [T, L, D]
        return _StubOutput(hidden_states=(hidden, hidden))  # 2 layers; last layer is taken

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@dataclass
class _StubPolicy:
    tokenizer: _StubTokenizer
    model: _StubModel


def _traj(task_id: str, n_turns: int, *, char_per_obs: int = 5, char_per_act: int = 4) -> Trajectory:
    turns = [
        TurnRecord(
            turn_idx=t,
            observation_text="o" * char_per_obs,
            action_text="a" * char_per_act,
        )
        for t in range(n_turns)
    ]
    return Trajectory(task_id=task_id, env_name="webshop", turns=turns, final_reward=0.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_embedder_returns_correct_shape_per_trajectory() -> None:
    """Vary T_i ∈ {1, 3, 5}; embedder returns [T_i, hidden_size] each time."""
    D = 8
    policy = _StubPolicy(tokenizer=_StubTokenizer(), model=_StubModel(hidden_size=D))
    embed = policy_hidden_state_embedder(policy, max_len_per_turn=64)

    for T_i in (1, 3, 5):
        out = embed(_traj("task-X", T_i))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (T_i, D)
        assert torch.isfinite(out).all()


def test_embedder_mean_pool_honors_padding_mask() -> None:
    """A 1-real-token + N-padded span pools to the value of the real token only.

    Construction:
    - Trajectory with 2 turns: turn 0 has a 1-char span (1 real token);
      turn 1 has a 5-char span (5 real tokens). Padding will fill turn 0
      to L=5 because the tokenizer pads to the longest span in the batch.
    - The stub's hidden state for token id k at position p is
      `(k + p*0.01) * 0.1 + d`. Padded positions (id=0) at positions
      [1..4] would contribute `(0 + 0.01..0.04)*0.1 + d` if the mask
      were ignored; the real-token-only mean must NOT include those.

    The embedder's mean-pool over L with the attention mask must produce
    the value computed from position 0 alone (within fp32 noise).
    """
    D = 4
    policy = _StubPolicy(tokenizer=_StubTokenizer(), model=_StubModel(hidden_size=D))
    embed = policy_hidden_state_embedder(policy)

    # turn 0: obs="x", action="" → span "x\n" → 2 chars; turn 1: obs="aaaaa",
    # action="bbbbb" → 11 chars. Use TurnRecord directly with custom strings.
    turn_short = TurnRecord(turn_idx=0, observation_text="x", action_text="")
    turn_long = TurnRecord(turn_idx=1, observation_text="aaaaa", action_text="bbbbb")
    traj = Trajectory(
        task_id="t",
        env_name="webshop",
        turns=[turn_short, turn_long],
        final_reward=0.0,
    )
    out = embed(traj)

    # Compute the expected pool for turn 0 (the short span) by replicating
    # the stub's math at the unmasked positions only.
    short_span = "x\n"  # f"{obs}\n{action}"
    n_real = len(short_span)  # 2 tokens
    # Tokenizer maps each char c → 1 + (ord(c) % 100). Hidden state at pos p
    # for token id k = (k + p*0.01) * 0.1 + d for d ∈ [0, D).
    ids = [1 + (ord(c) % 100) for c in short_span]
    feat_ramp = torch.arange(D, dtype=torch.float32)
    hiddens = []
    for p, k in enumerate(ids):
        hiddens.append((k + p * 0.01) * 0.1 + feat_ramp)
    expected = torch.stack(hiddens).mean(dim=0)

    assert out[0].shape == (D,)
    assert torch.allclose(out[0], expected, atol=1e-5), (
        f"mean-pool ignored mask: got {out[0].tolist()}, "
        f"expected {expected.tolist()}"
    )


def test_embedder_returns_cpu_fp32_regardless_of_model_dtype() -> None:
    """Even when the model's parameter dtype is fp64, the embedder returns
    CPU fp32 — the decomposer adapter casts to model dtype/device before forward.
    """
    D = 4
    policy = _StubPolicy(
        tokenizer=_StubTokenizer(),
        model=_StubModel(hidden_size=D, dtype=torch.float64),
    )
    embed = policy_hidden_state_embedder(policy)
    out = embed(_traj("task", n_turns=2))
    assert out.device.type == "cpu"
    assert out.dtype == torch.float32


def test_embedder_rejects_empty_trajectory() -> None:
    """The decomposer adapter normally filters empties; if a direct caller
    passes one through, the embedder raises rather than silently fabricating
    a hidden_size."""
    policy = _StubPolicy(tokenizer=_StubTokenizer(), model=_StubModel())
    embed = policy_hidden_state_embedder(policy)
    empty = Trajectory(
        task_id="empty", env_name="webshop", turns=[], final_reward=0.0
    )
    with pytest.raises(ValueError, match=r"no turns"):
        embed(empty)


def test_v10b_embedder_uses_hidden_states_not_logits_for_causal_lm() -> None:
    """v10b regression: the embedder must NOT silently consume LM-head
    logits as if they were hidden states.

    Background: v10's first attempt removed ``output_hidden_states=True``
    and tried to read ``outputs.last_hidden_state`` (then fell back to
    ``outputs[0]``). For HF ``AutoModelForCausalLM`` (the production
    policy model), ``outputs.last_hidden_state`` does NOT exist, and
    ``outputs[0]`` is the LM-head logits ([T, L, vocab_size=151936] for
    Qwen-1.5B). The embedder happily mean-pooled the logits and returned
    a [T, 151936] tensor, which then crashed inside the TurnRD model with
    ``forward: turn_embeds last dim 151936 != configured input_dim 1536``.
    Every TurnRD episode failed.

    This test simulates a CausalLM-shaped model: ``outputs.logits`` is
    populated (vocab_size 100), ``outputs.hidden_states`` is the proper
    [T, L, hidden_size=8] tuple, and ``outputs[0]`` aliases logits (the
    real HF ``CausalLMOutputWithPast`` ordering when ``loss`` is None).
    The embedder must return a tensor whose last dim is the
    HIDDEN_SIZE (8), not the VOCAB_SIZE (100).
    """

    @dataclass
    class _CausalLMOutput:
        logits: "torch.Tensor"
        hidden_states: tuple

        # Mimic HF's tuple-style indexing: outputs[0] = logits when loss
        # is None. This is exactly what tripped v10's first attempt up.
        def __getitem__(self, idx):
            if idx == 0:
                return self.logits
            raise IndexError(idx)

        # Note deliberately NO `last_hidden_state` field — that's only on
        # AutoModel, never on AutoModelForCausalLM.

    class _CausalLMStubModel(_StubModel):
        VOCAB_SIZE = 100  # would be 151936 in production

        def __init__(self, hidden_size: int = 8) -> None:
            super().__init__(hidden_size=hidden_size)
            self._vocab = self.VOCAB_SIZE

        def __call__(
            self,
            input_ids: "torch.Tensor",
            attention_mask: "torch.Tensor" | None = None,
            output_hidden_states: bool = False,
            **_: Any,
        ) -> _CausalLMOutput:
            T, L = input_ids.shape
            ids = input_ids.to(dtype=self._dtype)
            positions = torch.arange(L, dtype=self._dtype).unsqueeze(0)
            base = (ids + positions * 0.01) * 0.1
            feat_ramp = torch.arange(self.hidden_size, dtype=self._dtype)
            hidden = base.unsqueeze(-1) + feat_ramp  # [T, L, D]
            # Logits are independently shaped [T, L, vocab_size]. Filled
            # with arange so any accidental "use logits as hidden" path
            # would produce a tensor whose last dim != hidden_size.
            logits = torch.arange(
                T * L * self._vocab, dtype=self._dtype
            ).view(T, L, self._vocab)
            return _CausalLMOutput(
                logits=logits,
                hidden_states=(hidden, hidden),
            )

    HIDDEN = 8
    policy = _StubPolicy(
        tokenizer=_StubTokenizer(),
        model=_CausalLMStubModel(hidden_size=HIDDEN),
    )
    embed = policy_hidden_state_embedder(policy)
    out = embed(_traj("task", n_turns=3))
    # Bug regression: would have been (3, 100) if embedder consumed
    # outputs[0] (logits) instead of outputs.hidden_states[-1].
    assert out.shape == (3, HIDDEN), (
        f"embedder consumed wrong tensor: got shape {tuple(out.shape)}, "
        f"expected (3, {HIDDEN}). The last dim {out.shape[-1]} matches "
        f"vocab_size ({_CausalLMStubModel.VOCAB_SIZE}) ⇒ embedder is "
        "reading LM-head logits, not hidden states."
    )
