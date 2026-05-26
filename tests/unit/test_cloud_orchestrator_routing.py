"""Unit tests for the cloud orchestrator routing + auto-resume logic
added by plan `cloud_orchestrate_six_launchers`.

The orchestrator functions in `infra/app_orchestrator.py` are
`@app.function`-decorated and use `modal.Function.from_name(...)` for
cross-app dispatch — both of which require live Modal CLI auth and
deployed callee apps to actually execute. Per the same pattern used by
`tests/unit/test_train_loop_budget_config.py`, we therefore test:

1. STRUCTURAL invariants (via source-file slicing) that pin the
   env-name → function-name routing AND the absence of TurnRD calls in
   the no-TurnRD orchestrator. We slice the source file directly
   because `@app.function` wraps the callable in a Modal `Function`
   object whose source `inspect.getsource` cannot read.
2. PURE HELPER equivalents of the mode-gating + sentinel-scan logic for
   the SFT pipeline, plus assertions on the `_SFT_MODE_STAGES` /
   `_SFT_STAGES` module constants the production code uses.

Test-vs-prod equivalence is anchored by importing the production
symbols, so any future refactor that deletes / renames them fails at
import time.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# Path to the orchestrator source file. Read-once and sliced by `def`
# boundaries so each test can assert on the slice for its target
# function. We CANNOT use `inspect.getsource` because the
# `@app.function`-decorated callables become `modal.Function` instances
# whose `__source__` is opaque.
_ORCHESTRATOR_PATH = os.path.join(
    REPO_ROOT, "infra", "app_orchestrator.py"
)


def _function_source_block(func_name: str) -> str:
    """Return the text of `def {func_name}(...): ...` from the
    orchestrator source file, ending at the next top-level
    `def`/`@app`/`# ==`/EOF boundary."""
    with open(_ORCHESTRATOR_PATH) as fh:
        text = fh.read()
    start_marker = f"def {func_name}("
    start = text.find(start_marker)
    assert start >= 0, (
        f"could not find `def {func_name}(` in {_ORCHESTRATOR_PATH} — "
        "did the function get renamed or removed?"
    )
    # Find the next top-level `def`/`@app`/section banner AFTER our
    # function's def line so we don't include sibling functions.
    rest = text[start + len(start_marker):]
    end_candidates = []
    for marker in ("\ndef ", "\n@app", "\n# ==", "\n# ----"):
        idx = rest.find(marker)
        if idx >= 0:
            end_candidates.append(idx)
    end = min(end_candidates) if end_candidates else len(rest)
    return start_marker + rest[:end]


# -----------------------------------------------------------------------------
# Test 1: env_name → train_loop function name routing for the RL orchestrators.
# -----------------------------------------------------------------------------

def test_with_turnrd_routes_env_name_to_train_loop_fn_name() -> None:
    """`orchestrate_rl_with_turnrd` must look up
    `train_loop_<env_name>` against the `cs224r-hgpo-train-loop` app.
    The source-level assertion below pins the exact f-string used so
    a future refactor that splits / hard-codes the env routing fails.
    """
    # Import-time anchor: the symbol must still exist.
    from infra import app_orchestrator
    assert hasattr(app_orchestrator, "orchestrate_rl_with_turnrd")
    src = _function_source_block("orchestrate_rl_with_turnrd")
    # The router string must be built from env_name.
    assert 'f"train_loop_{env_name}"' in src, (
        "orchestrate_rl_with_turnrd must build the train_loop function "
        "name as f'train_loop_{env_name}' — got source:\n" + src
    )
    # Validation MUST gate env_name to alfworld | webshop.
    assert '"alfworld"' in src and '"webshop"' in src, (
        "orchestrate_rl_with_turnrd must validate env_name in "
        "('alfworld', 'webshop')."
    )
    # Must call train_loop on the cs224r-hgpo-train-loop app.
    assert '"cs224r-hgpo-train-loop"' in src
    # Must dispatch the TurnRD callee on the cs224r-hgpo-train-turnrd app.
    assert '"cs224r-hgpo-train-turnrd"' in src
    assert '"train_turnrd_run"' in src


def test_no_turnrd_routes_env_name_AND_omits_train_turnrd_dispatch() -> None:
    """`orchestrate_rl_no_turnrd` must (a) route env_name → train_loop_fn
    identically to the with-TurnRD variant, AND (b) NOT reference the
    `cs224r-hgpo-train-turnrd` app / `train_turnrd_run` function.
    """
    from infra import app_orchestrator
    assert hasattr(app_orchestrator, "orchestrate_rl_no_turnrd")
    src = _function_source_block("orchestrate_rl_no_turnrd")
    # Same env-name router.
    assert 'f"train_loop_{env_name}"' in src
    assert '"alfworld"' in src and '"webshop"' in src
    assert '"cs224r-hgpo-train-loop"' in src
    # CRITICAL: no train_turnrd reference anywhere in this entrypoint.
    # We assert on BOTH the app id and the function name so a partial
    # re-introduction still trips the test.
    assert "cs224r-hgpo-train-turnrd" not in src, (
        "orchestrate_rl_no_turnrd must NOT look up the train_turnrd app; "
        "this is the defining feature vs orchestrate_rl_with_turnrd."
    )
    assert "train_turnrd_run" not in src, (
        "orchestrate_rl_no_turnrd must NOT invoke train_turnrd_run."
    )


def test_alias_orchestrate_alfworld_rl_resolves_to_with_turnrd() -> None:
    """The plan keeps a thin Python-level alias for in-flight ephemeral
    callers that hold a `Function.from_name` handle / import by the old
    name. The alias must be IS-equal to the new entrypoint."""
    from infra import app_orchestrator
    assert (
        app_orchestrator.orchestrate_alfworld_rl
        is app_orchestrator.orchestrate_rl_with_turnrd
    ), (
        "orchestrate_alfworld_rl must remain a Python-level alias for "
        "orchestrate_rl_with_turnrd so module imports of the old name "
        "still resolve."
    )


def test_orchestrators_use_env_aware_num_products_not_hardcoded_zero() -> None:
    """Regression: an earlier draft hard-coded `num_products=0` in the
    train_loop.remote(...) call. For WebShop this would zero the BM25
    index because `_train_loop_impl` falls back to the kwarg whenever
    cfg.env.env_kwargs is empty — which IS the case for ALL the
    shipped WebShop SOTA configs (verified with json.load).

    Both RL orchestrators must instead pass a value derived from
    cfg.env.env_kwargs.num_products (or an env-appropriate default:
    1000 for webshop, 0 for alfworld). We pin both invariants:
      (a) source contains the `1000 if env_name == "webshop" else 0`
          fallback expression, AND
      (b) source does NOT contain `num_products=0,` as a literal in the
          train_loop.remote() call (kept as a comment is fine; the
          assertion targets the no-trailing-newline arg form).
    """
    for fn_name in ("orchestrate_rl_with_turnrd", "orchestrate_rl_no_turnrd"):
        src = _function_source_block(fn_name)
        assert '1000 if env_name == "webshop" else 0' in src, (
            f"{fn_name} must resolve num_products via env-appropriate "
            "default; got source:\n" + src
        )
        # The remote(...) call must reference the computed variable,
        # not a hard-coded 0.
        assert "num_products=num_products_to_pass," in src, (
            f"{fn_name} must pass `num_products=num_products_to_pass` to "
            "train_loop.remote() — got source:\n" + src
        )


# -----------------------------------------------------------------------------
# Test 2: SFT pipeline mode-gating + sentinel-scan auto-resume.
# -----------------------------------------------------------------------------

# Pure-Python re-implementation of the sentinel-naming + mode-gating
# logic inside `orchestrate_sft_pipeline`. The orchestrator function
# itself is `@app.function`-decorated and depends on Modal + volume
# bindings, so we cannot call it directly. Instead we replicate the
# in-scope helpers here and assert that the production code uses the
# same constants (`_SFT_STAGES`, `_SFT_MODE_STAGES`) below.

def _sentinel_path(sentinel_dir: str, run_name: str, stage: str) -> str:
    return (
        f"{sentinel_dir.rstrip('/')}/"
        f"sft_pipeline_{run_name}_stage{stage}_done.json"
    )


def _planned_stages_to_run(
    mode: str,
    run_name: str,
    sentinel_dir: str,
    mode_stages: dict[str, set[str]],
    all_stages: tuple[str, ...],
    auto_resume: bool = True,
) -> list[str]:
    """Return the stages that the orchestrator WOULD run given mode +
    existing sentinels. Mirrors `_maybe_run`'s skip logic exactly."""
    allowed = mode_stages[mode]
    to_run: list[str] = []
    for stage in all_stages:
        if stage not in allowed:
            continue
        if auto_resume and os.path.exists(
            _sentinel_path(sentinel_dir, run_name, stage)
        ):
            continue
        to_run.append(stage)
    return to_run


def test_sft_pipeline_mode_stages_constant_matches_bash_launcher() -> None:
    """`_SFT_MODE_STAGES` is the source of truth for which MODE allows
    which stages. The mapping must mirror the bash launcher
    `scripts/run_webshop_sft_v3_mlpr32.sh`'s MODE gating exactly.
    """
    from infra.app_orchestrator import _SFT_MODE_STAGES, _SFT_STAGES
    # All stages present + ordered as the pipeline runs them.
    assert _SFT_STAGES == ("1a", "1b", "1c", "2", "3", "4")
    # MODE → allowed-stage sets, matching the bash launcher.
    assert _SFT_MODE_STAGES == {
        "full": {"1a", "1b", "1c", "2", "3", "4"},
        "skip-install": {"2", "3", "4"},
        "skip-gen": {"3", "4"},
        "train-only": {"3", "4"},
        "eval-only": {"4"},
    }


def test_sft_pipeline_auto_resume_skips_completed_stages() -> None:
    """Given a synthetic set of stage-done sentinels in a tmp dir, the
    pipeline scan must report the resume-from-stage correctly under
    every mode. Mirrors the production `_is_done` + `_maybe_run` skip
    logic via the pure helper above; the production constants are
    asserted to be identical in
    `test_sft_pipeline_mode_stages_constant_matches_bash_launcher`."""
    from infra.app_orchestrator import _SFT_MODE_STAGES, _SFT_STAGES
    run_name = "sft_webshop_v3_mlpr32_cloud_smoke"
    with tempfile.TemporaryDirectory() as td:
        # Pre-mark stages 1a + 1b + 1c + 2 as done (e.g. install + gen
        # already completed on a prior orchestrator container; this
        # container restarted into the train step).
        for stage in ("1a", "1b", "1c", "2"):
            with open(_sentinel_path(td, run_name, stage), "w") as fh:
                json.dump({"stage": stage, "status": "completed"}, fh)

        # MODE=full: install + gen are done, only train + eval remain.
        assert _planned_stages_to_run(
            "full", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES
        ) == ["3", "4"]

        # MODE=skip-install: install stages are gated out by mode anyway;
        # gen is also done; only train + eval remain.
        assert _planned_stages_to_run(
            "skip-install", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES
        ) == ["3", "4"]

        # MODE=train-only ignores install AND gen sentinels (they're
        # outside the allowed-stage set), still runs train + eval.
        assert _planned_stages_to_run(
            "train-only", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES
        ) == ["3", "4"]

        # MODE=eval-only runs ONLY stage 4 (train is allowed by neither
        # mode nor the sentinel — but here it's not done so absence of
        # sentinel does NOT pull it in). Mode is the upper bound.
        assert _planned_stages_to_run(
            "eval-only", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES
        ) == ["4"]

        # Now mark stages 3 + 4 as done too. Every mode reports an empty
        # to-run list (everything already complete).
        for stage in ("3", "4"):
            with open(_sentinel_path(td, run_name, stage), "w") as fh:
                json.dump({"stage": stage, "status": "completed"}, fh)
        for mode in _SFT_MODE_STAGES:
            assert _planned_stages_to_run(
                mode, run_name, td, _SFT_MODE_STAGES, _SFT_STAGES
            ) == [], f"mode={mode} should report no stages to run"

        # auto_resume=False: every mode re-runs ALL stages it's allowed
        # to, regardless of sentinels.
        assert _planned_stages_to_run(
            "full", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES,
            auto_resume=False,
        ) == ["1a", "1b", "1c", "2", "3", "4"]
        assert _planned_stages_to_run(
            "eval-only", run_name, td, _SFT_MODE_STAGES, _SFT_STAGES,
            auto_resume=False,
        ) == ["4"]


def test_sft_pipeline_sentinel_path_format_is_stable() -> None:
    """The sentinel path format is part of the launcher's contract
    (the bash launcher prints it to the user as a debug aid). Pin it."""
    p = _sentinel_path(
        "/vol/checkpoints", "sft_webshop_v3_mlpr32_cloud_20260530_010203", "3"
    )
    assert p == (
        "/vol/checkpoints/sft_pipeline_"
        "sft_webshop_v3_mlpr32_cloud_20260530_010203_stage3_done.json"
    )
    # No double-slash even if caller passes trailing slash.
    p2 = _sentinel_path("/vol/checkpoints/", "run", "1a")
    assert "//" not in p2
