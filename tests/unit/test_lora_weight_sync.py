"""Unit tests for src.policy.weight_sync (pure-Python; no torch needed)."""

from __future__ import annotations

from src.policy.weight_sync import (
    canonicalize_lora_target_name,
    is_lora_param_name,
    plan_weight_sync,
    strip_peft_prefix,
)


# ---------- strip_peft_prefix ----------


def test_strip_peft_prefix_removes_wrapper() -> None:
    assert strip_peft_prefix(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    ) == "model.layers.0.self_attn.q_proj.base_layer.weight"


def test_strip_peft_prefix_passthrough_when_absent() -> None:
    assert strip_peft_prefix("model.layers.0.foo.weight") == "model.layers.0.foo.weight"


def test_strip_peft_prefix_idempotent() -> None:
    once = strip_peft_prefix("base_model.model.lm_head.weight")
    twice = strip_peft_prefix(once)
    assert once == twice == "lm_head.weight"


# ---------- is_lora_param_name ----------


def test_lora_a_b_detected() -> None:
    assert is_lora_param_name(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    )
    assert is_lora_param_name(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )


def test_base_layer_not_lora() -> None:
    assert not is_lora_param_name(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    )
    assert not is_lora_param_name("model.embed_tokens.weight")


def test_lora_embedding_detected() -> None:
    assert is_lora_param_name("base_model.model.model.embed_tokens.lora_embedding_A.default.weight")
    assert is_lora_param_name("base_model.model.model.embed_tokens.lora_embedding_B.default.weight")


# ---------- canonicalize_lora_target_name ----------


def test_canonicalize_strips_prefix_and_base_layer() -> None:
    assert canonicalize_lora_target_name(
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    ) == "model.layers.0.self_attn.q_proj.weight"


def test_canonicalize_handles_lm_head() -> None:
    # lm_head is rarely LoRA-wrapped, but PEFT still adds the prefix.
    assert canonicalize_lora_target_name("base_model.model.lm_head.weight") == "lm_head.weight"


def test_canonicalize_passthrough_for_unwrapped_key() -> None:
    assert canonicalize_lora_target_name("model.norm.weight") == "model.norm.weight"


# ---------- plan_weight_sync ----------


def test_plan_weight_sync_categorizes_three_groups() -> None:
    keys = [
        # 4 LoRA-wrapped attention projections × (base_layer + lora_A + lora_B) = 12 keys
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_A.default.weight",
        "base_model.model.model.layers.0.self_attn.k_proj.lora_B.default.weight",
        # an un-wrapped layer norm (passes through after prefix strip)
        "base_model.model.model.layers.0.input_layernorm.weight",
        # bookkeeping buffer that we should explicitly skip in sync
        "modules_to_save_buffer.something",
    ]
    out = plan_weight_sync(keys)
    assert len(out["lora_pair"]) == 4   # q_proj_A, q_proj_B, k_proj_A, k_proj_B
    assert len(out["passthrough"]) == 3 # 2 base_layer + 1 layernorm (has base_model prefix)
    assert len(out["skipped"]) == 1     # the bookkeeping buffer


def test_plan_weight_sync_empty_input() -> None:
    assert plan_weight_sync([]) == {"passthrough": [], "lora_pair": [], "skipped": []}


def test_plan_weight_sync_pure_lora_group() -> None:
    keys = [
        "base_model.model.x.lora_A.default.weight",
        "base_model.model.x.lora_B.default.weight",
        "base_model.model.x.lora_A.foo.weight",
    ]
    out = plan_weight_sync(keys)
    assert out["lora_pair"] == keys
    assert out["passthrough"] == []
    assert out["skipped"] == []
