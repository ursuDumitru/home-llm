"""Deterministic context selection and session/CLI integration without a GPU."""

import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from home_llm.chat_session import ChatSession
from home_llm.cli import run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.context_budget import ContextBudget, estimate_message_tokens
from home_llm.model_backend import ChatMessage, GenerationComplete, TextChunk
from home_llm.model_registry import ModelProfile, ModelRegistry
from home_llm.model_selection import ModelSelection
from home_llm.session_store import SessionStore, create_session


def budget_for_input(tokens: int) -> ContextBudget:
    """Build a budget with a known input allowance and real output/safety reserves."""
    return ContextBudget(tokens + 64 + 128, 64)


def fake_backend() -> Mock:
    """Record requests and return one short complete assistant turn."""
    backend = Mock()
    backend.stream_chat.side_effect = lambda messages: iter(
        (TextChunk("Yes"), GenerationComplete(1, 1, 1, 1, 1))
    )
    return backend


class ContextBudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = (
            ChatMessage("system", "Sys"),
            ChatMessage("user", "Old"),
            ChatMessage("assistant", "One"),
            ChatMessage("user", "New"),
            ChatMessage("assistant", "Two"),
        )
        self.prompt = ChatMessage("user", "Now")

    def test_estimator_counts_utf8_bytes_and_message_framing(self) -> None:
        for content, expected in (("abc", 9), ("abcd", 10), ("中文", 10), ("🙂", 10)):
            with self.subTest(content=content):
                self.assertEqual(
                    estimate_message_tokens(ChatMessage("user", content)), expected
                )

    def test_exact_fit_keeps_required_messages_and_whole_recent_turn(self) -> None:
        result = budget_for_input(36).select(self.history, self.prompt)
        self.assertEqual(
            result.messages, (self.history[0], *self.history[3:], self.prompt)
        )
        self.assertEqual(result.estimated_input_tokens, 36)
        self.assertEqual(result.input_budget, 36)
        self.assertEqual(result.selected_turns, 1)
        self.assertEqual(result.omitted_turns, 1)
        self.assertEqual(result, budget_for_input(36).select(self.history, self.prompt))

    def test_one_token_less_drops_the_entire_turn_and_larger_budget_keeps_all(
        self,
    ) -> None:
        smaller = budget_for_input(35).select(self.history, self.prompt)
        self.assertEqual(smaller.messages, (self.history[0], self.prompt))
        self.assertEqual(smaller.omitted_turns, 2)
        larger = budget_for_input(54).select(self.history, self.prompt)
        self.assertEqual(larger.messages, (*self.history, self.prompt))
        self.assertEqual(larger.omitted_turns, 0)

    def test_never_skips_a_recent_large_turn_to_include_an_older_small_turn(
        self,
    ) -> None:
        history = (*self.history[:-1], ChatMessage("assistant", "x" * 300))
        result = budget_for_input(36).select(history, self.prompt)
        self.assertEqual(result.selected_turns, 0)
        self.assertEqual(result.messages, (history[0], self.prompt))

    def test_prompt_and_system_must_fit_even_without_history(self) -> None:
        for history, prompt in (
            ((), ChatMessage("user", "x" * 300)),
            ((ChatMessage("system", "x" * 300),), self.prompt),
        ):
            with self.subTest(history=history):
                with self.assertRaisesRegex(ValueError, "Shorten.*No request was sent"):
                    budget_for_input(36).select(history, prompt)
        with self.assertRaisesRegex(ValueError, "only 0 remain"):
            budget_for_input(0).select((), self.prompt)

    def test_rejects_invalid_budgets_and_incomplete_history(self) -> None:
        for field in ("context_length", "max_output_tokens", "safety_margin"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        replace(budget_for_input(36), **{field: value})
        with self.assertRaisesRegex(ValueError, "smaller than"):
            ContextBudget(64, 64)
        with self.assertRaisesRegex(ValueError, "complete"):
            budget_for_input(36).select(self.history[:-1], self.prompt)
        with self.assertRaisesRegex(ValueError, "user message"):
            budget_for_input(36).select((), ChatMessage("assistant", "Bad"))


class BudgetedSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = fake_backend()
        self.history = (
            ChatMessage("system", "Sys"),
            ChatMessage("user", "Old"),
            ChatMessage("assistant", "One"),
            ChatMessage("user", "New"),
            ChatMessage("assistant", "Two"),
        )
        self.session = ChatSession.from_messages(
            self.backend, self.history, context_budget=budget_for_input(36)
        )

    def test_notice_precedes_backend_and_full_transcript_survives(self) -> None:
        reports = []

        def notice(report):
            self.backend.stream_chat.assert_not_called()
            reports.append(report)

        list(self.session.send("Now", on_context_selected=notice))
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0], self.session.last_context_selection)
        self.assertEqual(
            self.backend.stream_chat.call_args.args[0], reports[0].messages
        )
        self.assertEqual(self.session.messages[:5], self.history)
        self.assertEqual(len(self.session.messages), 7)
        self.assertEqual(self.session.context_start, 0)

    def test_oversized_prompt_leaves_state_unchanged_and_allows_retry(self) -> None:
        with self.assertRaisesRegex(ValueError, "No request was sent"):
            list(self.session.send("x" * 300))
        self.backend.stream_chat.assert_not_called()
        self.assertEqual(self.session.messages, self.history)
        self.assertIsNone(self.session.last_context_selection)
        list(self.session.send("Now"))
        self.backend.stream_chat.assert_called_once()

    def test_failure_cancellation_and_closed_stream_preserve_history(self) -> None:
        for error in (RuntimeError("offline"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):

                def fail(messages, error=error):
                    yield TextChunk("Partial")
                    raise error

                self.backend.stream_chat.side_effect = fail
                with self.assertRaises(type(error)):
                    list(self.session.send("Now"))
                self.assertEqual(self.session.messages, self.history)
                self.assertEqual(self.session.context_start, 0)

        stream = self.session.send("Now")
        next(stream)
        stream.close()
        self.assertEqual(self.session.messages, self.history)

    def test_clear_resets_last_selection_and_never_reintroduces_excluded_turns(
        self,
    ) -> None:
        list(self.session.send("Now"))
        self.session.clear()
        self.assertIsNone(self.session.last_context_selection)
        list(self.session.send("Hi"))
        self.assertEqual(
            self.backend.stream_chat.call_args.args[0],
            (self.history[0], ChatMessage("user", "Hi")),
        )
        self.assertEqual(self.session.last_context_selection.omitted_turns, 0)
        self.assertEqual(self.session.context_start, 3)
        self.assertEqual(len(self.session.messages), 9)


class ContextIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore(Path(directory.name))
        self.profile = ModelProfile(
            "local", "Local", "ollama", "example:small", True, 228, 64, None, None
        )
        self.registry = ModelRegistry(
            "local", "http://127.0.0.1:11434", (self.profile,)
        )
        self.backend = fake_backend()
        self.selection = ModelSelection(
            self.registry, self.profile, self.backend, store=self.store
        )
        self.selection.runtime = Mock()

    def test_save_load_uses_saved_budget_preserves_full_transcript_and_boundary(
        self,
    ) -> None:
        saved_profile = replace(self.profile, context_length=219)
        history = (
            ChatMessage("user", "Excluded"),
            ChatMessage("assistant", "Yes"),
            ChatMessage("user", "x" * 300),
            ChatMessage("assistant", "Yes"),
            ChatMessage("user", "New"),
            ChatMessage("assistant", "Yes"),
        )
        saved = self.store.save(create_session(saved_profile, history, context_start=1))
        with patch(
            "home_llm.model_selection.create_backend", return_value=self.backend
        ):
            self.selection.load(saved.session_id)
        self.assertEqual(self.selection.session.context_budget, ContextBudget(219, 64))
        list(self.selection.session.send("Now"))
        self.assertEqual(
            self.backend.stream_chat.call_args.args[0],
            (*history[-2:], ChatMessage("user", "Now")),
        )
        self.assertEqual(self.selection.session.last_context_selection.omitted_turns, 1)
        updated = self.selection.save()
        reloaded = self.store.load(updated.session_id)
        self.assertEqual(reloaded.messages[:6], history)
        self.assertEqual(reloaded.context_start, 1)
        self.assertEqual(reloaded.profile, saved_profile)

    def test_new_and_model_switch_apply_effective_limits(self) -> None:
        self.assertEqual(self.selection.session.context_budget, ContextBudget(228, 64))
        other = replace(
            self.profile, id="other", context_length=512, max_output_tokens=100
        )
        self.selection.registry = replace(self.registry, models=(self.profile, other))
        with patch(
            "home_llm.model_selection.create_backend", return_value=self.backend
        ):
            self.selection.select("other")
            self.assertEqual(
                self.selection.session.context_budget, ContextBudget(512, 100)
            )
            self.selection.new()
            self.assertEqual(
                self.selection.session.context_budget, ContextBudget(512, 100)
            )

    def test_cli_reports_omissions_budget_and_oversize_error_then_continues(
        self,
    ) -> None:
        # Three short completed turns exceed this deliberately small input budget.
        for prompt in ("One", "Two", "Tri"):
            list(self.selection.session.send(prompt))
        output = StringIO()
        inputs = iter(("Now", "/context", "x" * 300, "Hi", "/exit --discard"))
        result = run_cli(
            self.backend,
            ConsoleRenderer(stream=output, use_color=False),
            "Local",
            selection=self.selection,
            input_function=lambda prompt: next(inputs),
        )
        self.assertEqual(result, 0)
        rendered = " ".join(output.getvalue().split())
        for expected in (
            "omitted 2 older turns",
            "not sent to the model",
            "Template/safety reserve: 128",
            "Estimated input budget: 36",
            "not a guarantee",
            "Last attempted request: 1 history turns selected",
            "No request was sent",
            "Session ended.",
        ):
            self.assertIn(expected, rendered)
        self.assertEqual(self.backend.stream_chat.call_count, 5)
        self.assertEqual(len(self.selection.session.messages), 10)


if __name__ == "__main__":
    unittest.main()
