"""Unit tests for `infra.common.maybe_openai_secret`.

Verifies the C1 fix: the lazy `modal.Secret.from_name(...)` reference is
returned by default (so deploy fails clearly with Modal's own NotFoundError
when the secret is missing), and the `CS224R_SKIP_OPENAI_SECRET=1` env var
opts out so users without the secret can still deploy the baseline path.
"""

from __future__ import annotations

import logging

import modal
import pytest

from infra.common import (
    OPENAI_SECRET_NAME,
    OPENAI_SECRET_OPT_OUT_ENV,
    OPENAI_SECRET_REQUIRED_KEYS,
    maybe_openai_secret,
)


def test_default_returns_one_secret_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default behavior (env var unset) returns exactly one modal.Secret."""
    monkeypatch.delenv(OPENAI_SECRET_OPT_OUT_ENV, raising=False)
    secrets = maybe_openai_secret()
    assert len(secrets) == 1
    assert isinstance(secrets[0], modal.Secret)


def test_opt_out_env_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """CS224R_SKIP_OPENAI_SECRET=1 yields [] so deploy doesn't try to attach."""
    monkeypatch.setenv(OPENAI_SECRET_OPT_OUT_ENV, "1")
    assert maybe_openai_secret() == []


def test_opt_out_env_with_other_value_does_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the literal string "1" opts out; "true"/"yes"/"0" all keep secret."""
    for value in ("0", "true", "yes", "TRUE", ""):
        monkeypatch.setenv(OPENAI_SECRET_OPT_OUT_ENV, value)
        secrets = maybe_openai_secret()
        assert len(secrets) == 1, f"opt-out env={value!r} should NOT skip"


def test_opt_out_emits_warning_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Opting out logs a WARNING so the skip is visible in `modal run` output."""
    monkeypatch.setenv(OPENAI_SECRET_OPT_OUT_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="infra.common"):
        maybe_openai_secret()
    assert any(
        OPENAI_SECRET_OPT_OUT_ENV in record.message and record.levelname == "WARNING"
        for record in caplog.records
    ), f"expected WARNING containing {OPENAI_SECRET_OPT_OUT_ENV!r} in caplog, got: {caplog.records}"


def test_secret_constants_match_user_provisioned_name() -> None:
    """Sanity: the secret name constant matches the documented `modal secret create` name."""
    assert OPENAI_SECRET_NAME == "openai-secret"
    assert OPENAI_SECRET_REQUIRED_KEYS == ["OPENAI_API_KEY"]
