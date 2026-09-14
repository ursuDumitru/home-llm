"""Deterministic tests for performance benchmark coordination."""

import unittest
from collections.abc import Iterator, Sequence

from home_llm.benchmark import run_benchmark
from home_llm.gpu_monitor import (
    GpuReport,
    GpuSample,
)
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    StreamEvent,
    TextChunk,
)


class FakeBenchmarkBackend:
    """Return fixed metrics without contacting a model server."""

    def __init__(self) -> None:
        self.call_count = 0
        self.received_messages: list[tuple[ChatMessage, ...]] = []

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
    ) -> Iterator[StreamEvent]:
        self.call_count += 1
        self.received_messages.append(tuple(messages))

        load_duration_ns = 5_000_000_000 if self.call_count == 1 else 1_000_000

        yield TextChunk(text="1 2 ")
        yield TextChunk(text="3 4 5")
        yield GenerationComplete(
            prompt_tokens=20,
            generated_tokens=256,
            prompt_duration_ns=100_000_000,
            generation_duration_ns=2_000_000_000,
            total_duration_ns=2_100_000_000,
            load_duration_ns=load_duration_ns,
        )


class FakeGpuMonitor:
    """Record lifecycle calls and return fixed GPU samples."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> GpuReport:
        self.stopped = True

        return GpuReport(
            duration_seconds=2.1,
            samples=(
                GpuSample(
                    elapsed_seconds=0.0,
                    memory_used_mib=7000,
                    memory_total_mib=12227,
                    utilization_percent=10,
                ),
                GpuSample(
                    elapsed_seconds=2.0,
                    memory_used_mib=7200,
                    memory_total_mib=12227,
                    utilization_percent=90,
                ),
            ),
        )


class BenchmarkTests(unittest.TestCase):
    def test_runs_warmup_and_measured_generations(self) -> None:
        backend = FakeBenchmarkBackend()
        monitors: list[FakeGpuMonitor] = []

        def create_monitor() -> FakeGpuMonitor:
            monitor = FakeGpuMonitor()
            monitors.append(monitor)
            return monitor

        result = run_benchmark(
            backend,
            model_name="test-model",
            measured_runs=3,
            prompt="Fixed benchmark prompt.",
            gpu_monitor_factory=create_monitor,
        )

        self.assertEqual(backend.call_count, 4)
        self.assertEqual(len(result.measured_runs), 3)
        self.assertEqual(result.warmup_load_duration_seconds, 5.0)
        self.assertEqual(
            result.median_prompt_tokens_per_second,
            200.0,
        )
        self.assertEqual(
            result.median_generation_tokens_per_second,
            128.0,
        )
        self.assertEqual(
            result.median_total_duration_seconds,
            2.1,
        )
        self.assertEqual(result.minimum_generated_tokens, 256)
        self.assertEqual(result.maximum_generated_tokens, 256)
        self.assertTrue(result.outputs_are_identical)
        self.assertEqual(result.peak_gpu_memory_used_mib, 7200)
        self.assertEqual(result.peak_gpu_utilization_percent, 90)

        self.assertEqual(len(monitors), 4)
        self.assertTrue(all(monitor.started for monitor in monitors))
        self.assertTrue(all(monitor.stopped for monitor in monitors))

        self.assertEqual(
            backend.received_messages[0],
            (
                ChatMessage(
                    role="user",
                    content="Fixed benchmark prompt.",
                ),
            ),
        )

    def test_rejects_zero_measured_runs(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "measured benchmark runs must be greater than zero",
        ):
            run_benchmark(
                FakeBenchmarkBackend(),
                model_name="test-model",
                measured_runs=0,
            )


if __name__ == "__main__":
    unittest.main()
