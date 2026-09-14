"""Final acceptance checks before model-registry development."""

import unittest
from collections.abc import Iterator, Sequence
from functools import partial
from io import StringIO
from unittest.mock import patch

from home_llm.cli import run_cli
from home_llm.console_renderer import ConsoleRenderer
from home_llm.gpu_monitor import NvidiaSmiError
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    StreamEvent,
    TextChunk,
)
from home_llm.quality_evaluation import wait_for_ollama_model_unloaded
from home_llm.test_runner import GpuReportingTestResult


class RecoveringBackend:
    """Fail on the second turn, then accept a recovery request."""

    def __init__(self, failure_type: type[BaseException]) -> None:
        self.failure_type = failure_type
        self.requests: list[tuple[ChatMessage, ...]] = []

    def stream_chat(self, messages: Sequence[ChatMessage]) -> Iterator[StreamEvent]:
        self.requests.append(tuple(messages))

        if len(self.requests) == 2:
            yield TextChunk(text="Partial response")
            raise self.failure_type("Simulated interruption")

        yield TextChunk(text="Complete response")
        yield GenerationComplete(
            prompt_tokens=10,
            generated_tokens=2,
            prompt_duration_ns=100_000_000,
            generation_duration_ns=100_000_000,
            total_duration_ns=200_000_000,
        )


class ReleaseCheckpointTests(unittest.TestCase):
    def test_cli_preserves_history_and_recovers_after_interruption(self) -> None:
        for failure_type in (RuntimeError, KeyboardInterrupt):
            with self.subTest(failure_type=failure_type.__name__):
                backend = RecoveringBackend(failure_type)
                output = StringIO()
                inputs = iter(["First prompt", "Interrupted prompt", "Retry", "/exit"])

                exit_code = run_cli(
                    backend=backend,
                    renderer=ConsoleRenderer(stream=output, use_color=False),
                    model_name="fake-model",
                    input_function=lambda prompt, inputs=inputs: next(inputs),
                )

                previous_turn = (
                    ChatMessage(role="user", content="First prompt"),
                    ChatMessage(role="assistant", content="Complete response"),
                )
                self.assertEqual(exit_code, 0)
                self.assertEqual(len(backend.requests), 3)
                self.assertEqual(
                    backend.requests[1],
                    (
                        *previous_turn,
                        ChatMessage(role="user", content="Interrupted prompt"),
                    ),
                )
                self.assertEqual(
                    backend.requests[2],
                    (*previous_turn, ChatMessage(role="user", content="Retry")),
                )

                expected_error = (
                    "Generation cancelled."
                    if failure_type is KeyboardInterrupt
                    else "Simulated interruption"
                )
                self.assertIn(expected_error, output.getvalue())
                self.assertIn("Session ended.", output.getvalue())

    def test_gpu_failure_is_visible_without_failing_successful_test(self) -> None:
        output = StringIO()
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        runner = unittest.TextTestRunner(
            stream=output,
            verbosity=2,
            resultclass=partial(
                GpuReportingTestResult,
                gpu_sample_interval=0.1,
            ),
        )

        with patch(
            "home_llm.gpu_monitor.read_nvidia_smi",
            side_effect=NvidiaSmiError("Simulated GPU query failure"),
        ):
            result = runner.run(suite)

        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful())
        self.assertIn("GPU monitoring unavailable", output.getvalue())
        self.assertIn("Simulated GPU query failure", output.getvalue())

    def test_unload_wait_times_out_when_model_stays_loaded(self) -> None:
        with (
            patch(
                "home_llm.quality_evaluation.list_running_ollama_models",
                return_value={"test-model"},
            ),
            patch(
                "home_llm.quality_evaluation.time.monotonic",
                side_effect=[0.0, 0.1, 1.1],
            ),
            patch("home_llm.quality_evaluation.time.sleep"),
            self.assertRaisesRegex(
                RuntimeError,
                "did not unload 'test-model' within 1.0 seconds",
            ),
        ):
            wait_for_ollama_model_unloaded(
                "test-model",
                timeout_seconds=1.0,
                polling_interval_seconds=0.1,
            )


if __name__ == "__main__":
    unittest.main()
