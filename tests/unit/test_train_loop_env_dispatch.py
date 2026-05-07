"""Unit test for `infra/app_train_loop.py::_resolve_env_bindings`.

Asserts that the env-name dispatch resolves to the correct
`(adapter_class, prompt_renderer, action_parser)` triple for each
supported env. Pure-Python (no Modal client needed): we import the
helper directly. Regression guard so future env additions don't
silently break dispatch.

Note: `infra/app_train_loop.py` does `import modal` at the top. Modal is
available on the dev box (`pip install modal`); on CI we'd skip via
`pytest.importorskip("modal")`. We do the same here for hermeticity.
"""
from __future__ import annotations

import unittest

import pytest

modal = pytest.importorskip("modal")  # noqa: F401  — used transitively by the import below


from infra.app_train_loop import _resolve_env_bindings  # noqa: E402


class TestEnvDispatch(unittest.TestCase):
    def test_webshop_dispatch(self) -> None:
        from src.envs.prompts.react_webshop import (
            parse_react_action as webshop_parse,
            render_webshop_turn_prompt,
        )
        from src.envs.webshop_adapter import WebShopAdapter

        adapter_cls, renderer, parser = _resolve_env_bindings("webshop")
        self.assertIs(adapter_cls, WebShopAdapter)
        self.assertIs(renderer, render_webshop_turn_prompt)
        self.assertIs(parser, webshop_parse)

    def test_alfworld_dispatch(self) -> None:
        from src.envs.alfworld_adapter import ALFWorldAdapter
        from src.envs.prompts.react_alfworld import (
            parse_react_action as alfworld_parse,
            render_alfworld_turn_prompt,
        )

        adapter_cls, renderer, parser = _resolve_env_bindings("alfworld")
        self.assertIs(adapter_cls, ALFWorldAdapter)
        self.assertIs(renderer, render_alfworld_turn_prompt)
        self.assertIs(parser, alfworld_parse)

    def test_unknown_env_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _resolve_env_bindings("bogus_env")
        self.assertIn("bogus_env", str(ctx.exception))
        self.assertIn("webshop", str(ctx.exception))
        self.assertIn("alfworld", str(ctx.exception))

    def test_webshop_and_alfworld_parsers_are_distinct(self) -> None:
        """Each env exposes its own `parse_react_action` import path so
        future per-env divergence (e.g. AlfWorld-specific verb
        whitelisting) doesn't entangle them. Verify the two are
        independent function objects even though they currently share
        a body."""
        _, _, webshop_parser = _resolve_env_bindings("webshop")
        _, _, alfworld_parser = _resolve_env_bindings("alfworld")
        self.assertIsNot(webshop_parser, alfworld_parser)


if __name__ == "__main__":
    unittest.main()
