"""Deterministic tests for the Ollama backend HTTP translation."""

import json
import unittest
from unittest.mock import patch

from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    TextChunk,
)
from home_llm.ollama_backend import OllamaBackend


class FakeHttpResponse:
    """Minimal iterable context manager returned by mocked urlopen."""

    def __init__(self, response_lines: list[bytes]) -> None:
        self._response_lines = response_lines

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> bool:
        return False

    def __iter__(self):
        return iter(self._response_lines)


class OllamaBackendTests(unittest.TestCase):
    @patch("home_llm.ollama_backend.urlopen")
    def test_sends_deterministic_options_and_returns_metrics(
        self,
        urlopen_mock,
    ) -> None:
        response_document = {
            "message": {
                "role": "assistant",
                "content": "BENCHMARK_OK",
            },
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 10,
            "prompt_eval_duration": 100_000_000,
            "eval_duration": 500_000_000,
            "total_duration": 1_000_000_000,
            "load_duration": 250_000_000,
        }

        response_line = json.dumps(response_document).encode("utf-8") + b"\n"
        urlopen_mock.return_value = FakeHttpResponse([response_line])

        backend = OllamaBackend(
            model="test-model",
            context_length=4096,
            max_output_tokens=256,
            timeout_seconds=30,
            temperature=0.0,
            seed=42,
        )

        events = list(
            backend.stream_chat(
                (
                    ChatMessage(
                        role="user",
                        content="Run the benchmark.",
                    ),
                )
            )
        )

        self.assertEqual(
            events[0],
            TextChunk(text="BENCHMARK_OK"),
        )
        self.assertIsInstance(events[1], GenerationComplete)

        completion = events[1]
        assert isinstance(completion, GenerationComplete)

        self.assertEqual(completion.load_duration_ns, 250_000_000)
        self.assertEqual(completion.generated_tokens, 10)

        request = urlopen_mock.call_args.args[0]
        request_payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(
            request_payload["options"],
            {
                "num_ctx": 4096,
                "num_predict": 256,
                "temperature": 0.0,
                "seed": 42,
            },
        )
        self.assertFalse(request_payload["think"])
        self.assertTrue(request_payload["stream"])

    def test_rejects_negative_temperature(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Temperature must not be negative",
        ):
            OllamaBackend(
                model="test-model",
                temperature=-0.1,
            )

    def test_rejects_negative_seed(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Random seed must not be negative",
        ):
            OllamaBackend(
                model="test-model",
                seed=-1,
            )


if __name__ == "__main__":
    unittest.main()
