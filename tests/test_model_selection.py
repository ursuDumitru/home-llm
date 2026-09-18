"""Focused checks for runtime status and safe interactive model selection."""

import json
import unittest
from dataclasses import replace
from io import BytesIO, StringIO
from unittest.mock import Mock, patch

from home_llm.chat_session import ChatSession
from home_llm.cli import run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import GenerationComplete, TextChunk
from home_llm.model_registry import ModelProfile, ModelRegistry
from home_llm.model_selection import ModelSelection
from home_llm.ollama_runtime import OllamaRuntime


class ModelSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = ModelProfile(
            "first", "First", "ollama", "first:small", True, 4096, 64, None, None
        )
        self.second = replace(
            self.first,
            id="second",
            display_name="Second",
            model_name="second:large",
            max_output_tokens=128,
        )
        disabled = replace(self.first, id="disabled", enabled=False)
        self.registry = ModelRegistry(
            "first", "http://127.0.0.1:11434", (self.first, self.second, disabled)
        )
        self.backend = Mock()
        self.backend.stream_chat.side_effect = lambda messages: iter(
            (
                TextChunk(text="Hello"),
                GenerationComplete(1, 1, 1, 1, 1),
            )
        )
        self.selection = ModelSelection(self.registry, self.first, self.backend)
        self.selection.runtime = Mock()
        self.selection.runtime.model_names.side_effect = lambda **kwargs: frozenset(
            {"first:small", "second:large"}
        )

    def test_successful_switch_uses_profile_settings_and_empty_history(self) -> None:
        with patch("home_llm.model_selection.create_backend") as factory:
            factory.return_value = self.backend
            self.selection.select("second")
        factory.assert_called_once_with(
            self.second, ollama_base_url=self.registry.ollama_base_url
        )
        self.selection.runtime.prepare.assert_called_once_with(self.second)
        self.assertEqual(self.selection.profile, self.second)
        self.assertEqual(self.selection.session.messages, ())

    def test_unknown_disabled_and_nonempty_selections_preserve_state(self) -> None:
        for target, expected in (("missing", "Unknown"), ("disabled", "disabled")):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, expected):
                    self.selection.select(target)

        list(self.selection.session.send("Keep this"))
        original = self.selection.session
        messages = original.messages
        with self.assertRaisesRegex(ValueError, "empty conversation"):
            self.selection.select("second")

        self.assertIs(self.selection.session, original)
        self.assertEqual(original.messages, messages)
        self.assertEqual(self.selection.profile, self.first)
        self.selection.runtime.prepare.assert_not_called()

    def test_system_prompt_is_not_silently_reused(self) -> None:
        self.selection.session = ChatSession(
            self.backend, system_prompt="Private rules"
        )
        with self.assertRaisesRegex(ValueError, "system prompt"):
            self.selection.select("second")
        self.selection.runtime.prepare.assert_not_called()

    def test_failed_or_cancelled_load_preserves_selection(self) -> None:
        original = self.selection.session
        for error in (
            ValueError("not installed"),
            RuntimeError("load failed"),
            KeyboardInterrupt(),
        ):
            with self.subTest(error=type(error).__name__):
                self.selection.runtime.prepare.side_effect = error
                with self.assertRaises(type(error)):
                    self.selection.select("second")
                self.assertIs(self.selection.session, original)
                self.assertEqual(self.selection.profile, self.first)
                self.assertEqual(original.messages, ())

    def test_partial_status_failure_preserves_known_information(self) -> None:
        self.selection.runtime.model_names.side_effect = (
            frozenset({"first:small"}),
            RuntimeError("service unavailable"),
        )
        rows, errors = self.selection.list_models()

        self.assertTrue(rows[0].installed)
        self.assertFalse(rows[1].installed)
        self.assertIsNone(rows[0].loaded)
        self.assertEqual(len(errors), 1)
        self.assertIn("Loaded status unavailable", errors[0])

    def test_cli_lists_switches_rejects_history_then_starts_new(self) -> None:
        inputs = iter(
            (
                "/models",
                "/model second",
                "Hi",
                "/model first",
                "/new --discard",
                "/model first",
                "/model",
                "/exit",
            )
        )
        output = StringIO()
        second_backend = Mock()
        second_backend.stream_chat.side_effect = self.backend.stream_chat.side_effect

        with patch("home_llm.model_selection.create_backend") as factory:
            factory.side_effect = (second_backend, self.backend, self.backend)
            result = run_cli(
                self.backend,
                ConsoleRenderer(stream=output, use_color=False),
                "First",
                selection=self.selection,
                input_function=lambda prompt: next(inputs),
            )

        self.assertEqual(result, 0)
        second_backend.stream_chat.assert_called_once()
        self.backend.stream_chat.assert_not_called()
        self.assertEqual(self.selection.profile, self.first)
        self.assertEqual(self.selection.runtime.prepare.call_count, 2)
        self.assertIn("Configured models", output.getvalue())
        self.assertIn("empty conversation", output.getvalue())
        self.assertIn("output limit=128", output.getvalue())
        self.assertIn("Active model: First [first]", output.getvalue())


class OllamaRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = OllamaRuntime("http://127.0.0.1:11434")
        self.profile = ModelProfile(
            "test", "Test", "ollama", "test", True, 4096, 64, None, None
        )

    def test_inventory_endpoints_and_default_tag(self) -> None:
        for loaded, path in ((False, "/api/tags"), (True, "/api/ps")):
            with (
                self.subTest(loaded=loaded),
                patch("home_llm.ollama_runtime.urlopen") as http,
            ):
                http.return_value = BytesIO(b'{"models":[{"name":"test:latest"}]}')
                self.assertEqual(
                    self.runtime.model_names(loaded=loaded), {"test:latest"}
                )
                request = http.call_args.args[0]
                self.assertEqual(request.full_url, f"http://127.0.0.1:11434{path}")
                self.assertEqual(request.get_method(), "GET")
                self.assertEqual(http.call_args.kwargs["timeout"], 5.0)

    def test_load_has_no_chat_content_and_uses_profile_context(self) -> None:
        with patch("home_llm.ollama_runtime.urlopen") as http:
            http.side_effect = (
                BytesIO(b'{"models":[{"name":"test:latest"}]}'),
                BytesIO(b'{"done":true,"done_reason":"load"}'),
            )
            self.runtime.prepare(self.profile)

        request = http.call_args.args[0]
        payload = json.loads(request.data)
        self.assertTrue(request.full_url.endswith("/api/chat"))
        self.assertEqual(payload["messages"], [])
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"], {"num_ctx": 4096})
        self.assertEqual(http.call_args.kwargs["timeout"], 180.0)

    def test_missing_model_never_attempts_load_or_download(self) -> None:
        with patch("home_llm.ollama_runtime.urlopen") as http:
            http.return_value = BytesIO(b'{"models":[]}')
            with self.assertRaisesRegex(ValueError, "not installed"):
                self.runtime.prepare(self.profile)
            http.assert_called_once()

    def test_invalid_inventory_and_network_failure_are_explicit(self) -> None:
        for payload in (b"bad json", b"[]", b"{}", b'{"models":[{}]}'):
            with (
                self.subTest(payload=payload),
                patch("home_llm.ollama_runtime.urlopen") as http,
            ):
                http.return_value = BytesIO(payload)
                with self.assertRaises(RuntimeError):
                    self.runtime.model_names()

        with patch("home_llm.ollama_runtime.urlopen", side_effect=TimeoutError):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                self.runtime.model_names()

    def test_incomplete_load_is_not_success(self) -> None:
        with patch.object(self.runtime, "_request") as request:
            request.side_effect = ({"models": [{"name": "test"}]}, {"done": False})
            with self.assertRaisesRegex(RuntimeError, "did not confirm"):
                self.runtime.prepare(self.profile)

    def test_remote_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "local server URL"):
            OllamaRuntime("https://example.com")


if __name__ == "__main__":
    unittest.main()
