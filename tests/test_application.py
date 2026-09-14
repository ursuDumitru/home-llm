"""Deterministic tests for the HomeLLM application layers."""

import argparse
import unittest
from collections.abc import Iterator, Sequence
from io import StringIO

from home_llm.chat_session import ChatSession
from home_llm.cli import (
    local_model_name,
    positive_integer,
    run_cli,
)
from home_llm.console_renderer import ConsoleRenderer
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    StreamEvent,
    TextChunk,
)


class FakeBackend:
    """Backend that returns a fixed successful response."""

    def __init__(self) -> None:
        self.received_messages: tuple[ChatMessage, ...] = ()

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        self.received_messages = tuple(messages)

        yield TextChunk(text="Hello")
        yield TextChunk(text=" from the fake model.")
        yield GenerationComplete(
            prompt_tokens=10,
            generated_tokens=5,
            prompt_duration_ns=100_000_000,
            generation_duration_ns=500_000_000,
            total_duration_ns=1_000_000_000,
        )


class FailingBackend:
    """Backend that fails after producing a partial response."""

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        yield TextChunk(text="Partial")
        raise RuntimeError("Simulated backend failure.")


class GenerationCompleteTests(unittest.TestCase):
    def test_calculates_token_speeds(self) -> None:
        completion = GenerationComplete(
            prompt_tokens=20,
            generated_tokens=10,
            prompt_duration_ns=500_000_000,
            generation_duration_ns=2_000_000_000,
            total_duration_ns=2_500_000_000,
        )

        self.assertEqual(
            completion.prompt_tokens_per_second,
            40.0,
        )
        self.assertEqual(
            completion.generation_tokens_per_second,
            5.0,
        )


class ChatSessionTests(unittest.TestCase):
    def test_records_successful_conversation(self) -> None:
        backend = FakeBackend()
        session = ChatSession(
            backend,
            system_prompt="Answer concisely.",
        )

        events = list(session.send("Hello"))

        self.assertEqual(len(events), 3)
        self.assertEqual(
            backend.received_messages,
            (
                ChatMessage(
                    role="system",
                    content="Answer concisely.",
                ),
                ChatMessage(role="user", content="Hello"),
            ),
        )
        self.assertEqual(
            session.messages,
            (
                ChatMessage(
                    role="system",
                    content="Answer concisely.",
                ),
                ChatMessage(role="user", content="Hello"),
                ChatMessage(
                    role="assistant",
                    content="Hello from the fake model.",
                ),
            ),
        )

    def test_rolls_back_failed_turn(self) -> None:
        session = ChatSession(
            FailingBackend(),
            system_prompt="Answer concisely.",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Simulated backend failure",
        ):
            list(session.send("This request will fail."))

        self.assertEqual(
            session.messages,
            (
                ChatMessage(
                    role="system",
                    content="Answer concisely.",
                ),
            ),
        )

    def test_clear_preserves_system_prompt(self) -> None:
        session = ChatSession(
            FakeBackend(),
            system_prompt="Answer concisely.",
        )

        list(session.send("Hello"))
        session.clear()

        self.assertEqual(
            session.messages,
            (
                ChatMessage(
                    role="system",
                    content="Answer concisely.",
                ),
            ),
        )


class ConsoleRendererTests(unittest.TestCase):
    def test_renders_plain_deterministic_output(self) -> None:
        output = StringIO()
        renderer = ConsoleRenderer(
            stream=output,
            use_color=False,
        )

        renderer.show_prompt()
        renderer.begin_answer()
        renderer.render_event(TextChunk(text="[bold]literal model text[/bold]"))
        renderer.render_event(
            GenerationComplete(
                prompt_tokens=10,
                generated_tokens=5,
                prompt_duration_ns=100_000_000,
                generation_duration_ns=500_000_000,
                total_duration_ns=1_000_000_000,
            )
        )

        expected = (
            "PROMPT > \n"
            "ANSWER\n"
            "[bold]literal model text[/bold]\n\n"
            "[prompt: 100.0 tok/s | generation: 10.0 tok/s"
            " | output: 5 tokens | total: 1.00 s]\n" + "─" * 72 + "\n"
        )

        self.assertEqual(output.getvalue(), expected)
        self.assertNotIn("\033[", output.getvalue())


class CliTests(unittest.TestCase):
    def test_commands_and_model_response(self) -> None:
        inputs = iter(
            [
                "/help",
                "/model",
                "/clear",
                "Hello",
                "/unknown",
                "/exit",
            ]
        )

        backend = FakeBackend()
        output = StringIO()
        renderer = ConsoleRenderer(
            stream=output,
            use_color=False,
        )

        exit_code = run_cli(
            backend=backend,
            renderer=renderer,
            model_name="test-model",
            input_function=lambda prompt: next(inputs),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            backend.received_messages,
            (ChatMessage(role="user", content="Hello"),),
        )

        rendered = output.getvalue()

        self.assertIn("Available commands:", rendered)
        self.assertIn("Active model: test-model", rendered)
        self.assertIn("Conversation history cleared.", rendered)
        self.assertIn("Hello from the fake model.", rendered)
        self.assertIn("Unknown command '/unknown'", rendered)
        self.assertIn("Session ended.", rendered)


class CommandLineValidationTests(unittest.TestCase):
    def test_accepts_positive_integer(self) -> None:
        self.assertEqual(positive_integer("4096"), 4096)

    def test_rejects_nonpositive_integer(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_integer("0")

    def test_accepts_local_model(self) -> None:
        self.assertEqual(
            local_model_name(" qwen3.5:4b "),
            "qwen3.5:4b",
        )

    def test_rejects_cloud_model(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            local_model_name("gpt-oss:120b-cloud")


if __name__ == "__main__":
    unittest.main()
