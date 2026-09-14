"""Repeatable performance benchmark for local model backends."""

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from statistics import fmean, median

from rich.console import Console
from rich.table import Table

from home_llm.cli import local_model_name, positive_integer
from home_llm.gpu_monitor import (
    GpuMonitor,
    GpuReport,
    GpuSample,
    NvidiaSmiGpuMonitor,
)
from home_llm.model_backend import (
    ChatMessage,
    GenerationComplete,
    ModelBackend,
    TextChunk,
)
from home_llm.ollama_backend import OllamaBackend

DEFAULT_CONTEXT_LENGTH = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_MEASURED_RUNS = 3
DEFAULT_GPU_SAMPLE_INTERVAL = 0.1
BENCHMARK_SEED = 42
BENCHMARK_TEMPERATURE = 0.0

BENCHMARK_PROMPT = (
    "Continue this integer sequence: 1 2 3 4 5. "
    "Return only consecutive integers separated by single spaces. "
    "Continue until the response reaches its output-token limit."
)

type GpuMonitorFactory = Callable[[], GpuMonitor]


@dataclass(frozen=True, slots=True)
class BenchmarkMeasurement:
    """Metrics and output from one model generation."""

    answer_text: str
    completion: GenerationComplete
    wall_duration_seconds: float
    gpu_report: GpuReport | None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Warm-up and measured results for one model."""

    model_name: str
    warmup: BenchmarkMeasurement
    measured_runs: tuple[BenchmarkMeasurement, ...]

    @property
    def all_measurements(self) -> tuple[BenchmarkMeasurement, ...]:
        """Return the warm-up followed by all measured runs."""

        return (self.warmup, *self.measured_runs)

    @property
    def median_prompt_tokens_per_second(self) -> float:
        return float(
            median(
                measurement.completion.prompt_tokens_per_second
                for measurement in self.measured_runs
            )
        )

    @property
    def median_generation_tokens_per_second(self) -> float:
        return float(
            median(
                measurement.completion.generation_tokens_per_second
                for measurement in self.measured_runs
            )
        )

    @property
    def median_total_duration_seconds(self) -> float:
        return float(
            median(
                measurement.completion.total_duration_ns / 1_000_000_000
                for measurement in self.measured_runs
            )
        )

    @property
    def median_wall_duration_seconds(self) -> float:
        return float(
            median(
                measurement.wall_duration_seconds for measurement in self.measured_runs
            )
        )

    @property
    def warmup_load_duration_seconds(self) -> float:
        return self.warmup.completion.load_duration_ns / 1_000_000_000

    @property
    def minimum_generated_tokens(self) -> int:
        return min(
            measurement.completion.generated_tokens
            for measurement in self.measured_runs
        )

    @property
    def maximum_generated_tokens(self) -> int:
        return max(
            measurement.completion.generated_tokens
            for measurement in self.measured_runs
        )

    @property
    def outputs_are_identical(self) -> bool:
        return len({measurement.answer_text for measurement in self.measured_runs}) == 1

    @property
    def gpu_samples(self) -> tuple[GpuSample, ...]:
        """Combine samples from warm-up and measured runs."""

        return tuple(
            sample
            for measurement in self.all_measurements
            if measurement.gpu_report is not None
            for sample in measurement.gpu_report.samples
        )

    @property
    def peak_gpu_memory_used_mib(self) -> int | None:
        if not self.gpu_samples:
            return None

        return max(sample.memory_used_mib for sample in self.gpu_samples)

    @property
    def total_gpu_memory_mib(self) -> int | None:
        if not self.gpu_samples:
            return None

        return self.gpu_samples[0].memory_total_mib

    @property
    def average_gpu_utilization_percent(self) -> float | None:
        if not self.gpu_samples:
            return None

        return fmean(sample.utilization_percent for sample in self.gpu_samples)

    @property
    def peak_gpu_utilization_percent(self) -> int | None:
        if not self.gpu_samples:
            return None

        return max(sample.utilization_percent for sample in self.gpu_samples)

    @property
    def gpu_errors(self) -> tuple[str, ...]:
        return tuple(
            measurement.gpu_report.error
            for measurement in self.all_measurements
            if (
                measurement.gpu_report is not None
                and measurement.gpu_report.error is not None
            )
        )


def positive_float(value: str) -> float:
    """Parse a command-line floating-point number greater than zero."""

    try:
        parsed_value = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a number, received {value!r}."
        ) from error

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a value greater than zero, received {parsed_value}."
        )

    return parsed_value


def measure_generation(
    backend: ModelBackend,
    *,
    prompt: str,
    gpu_monitor_factory: GpuMonitorFactory | None = None,
) -> BenchmarkMeasurement:
    """Measure one independent generation request."""

    gpu_monitor = gpu_monitor_factory() if gpu_monitor_factory is not None else None

    if gpu_monitor is not None:
        gpu_monitor.start()

    gpu_report: GpuReport | None = None
    answer_fragments: list[str] = []
    completion: GenerationComplete | None = None
    started_at = time.perf_counter()

    try:
        for event in backend.stream_chat(
            (
                ChatMessage(
                    role="user",
                    content=prompt,
                ),
            )
        ):
            if isinstance(event, TextChunk):
                answer_fragments.append(event.text)
                continue

            if isinstance(event, GenerationComplete):
                if completion is not None:
                    raise RuntimeError(
                        "The backend returned multiple completion events."
                    )

                completion = event
                continue

            raise TypeError(
                "The backend returned an unsupported benchmark event: "
                f"{type(event).__name__}."
            )

        wall_duration_seconds = time.perf_counter() - started_at
    finally:
        if gpu_monitor is not None:
            gpu_report = gpu_monitor.stop()

    answer_text = "".join(answer_fragments)

    if not answer_text.strip():
        raise RuntimeError("The benchmark generation returned no assistant text.")

    if completion is None:
        raise RuntimeError("The benchmark generation returned no completion metrics.")

    return BenchmarkMeasurement(
        answer_text=answer_text,
        completion=completion,
        wall_duration_seconds=wall_duration_seconds,
        gpu_report=gpu_report,
    )


def run_benchmark(
    backend: ModelBackend,
    *,
    model_name: str,
    measured_runs: int,
    prompt: str = BENCHMARK_PROMPT,
    gpu_monitor_factory: GpuMonitorFactory | None = None,
) -> BenchmarkResult:
    """Run one warm-up followed by measured generations."""

    if measured_runs <= 0:
        raise ValueError(
            "The number of measured benchmark runs must be greater than zero."
        )

    warmup = measure_generation(
        backend,
        prompt=prompt,
        gpu_monitor_factory=gpu_monitor_factory,
    )

    measurements = tuple(
        measure_generation(
            backend,
            prompt=prompt,
            gpu_monitor_factory=gpu_monitor_factory,
        )
        for _ in range(measured_runs)
    )

    return BenchmarkResult(
        model_name=model_name,
        warmup=warmup,
        measured_runs=measurements,
    )


def render_benchmark_result(
    result: BenchmarkResult,
    console: Console,
) -> None:
    """Render a benchmark summary as a terminal table."""

    table = Table(
        title=f"Benchmark: {result.model_name}",
        show_header=False,
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Result", justify="right")

    table.add_row(
        "Measured runs",
        str(len(result.measured_runs)),
    )
    table.add_row(
        "Warm-up model load",
        f"{result.warmup_load_duration_seconds:.3f} s",
    )
    table.add_row(
        "Median prompt speed",
        f"{result.median_prompt_tokens_per_second:.2f} tok/s",
    )
    table.add_row(
        "Median generation speed",
        f"{result.median_generation_tokens_per_second:.2f} tok/s",
    )
    table.add_row(
        "Median Ollama total",
        f"{result.median_total_duration_seconds:.3f} s",
    )
    table.add_row(
        "Median client wall time",
        f"{result.median_wall_duration_seconds:.3f} s",
    )
    table.add_row(
        "Generated-token range",
        (f"{result.minimum_generated_tokens}-{result.maximum_generated_tokens}"),
    )
    table.add_row(
        "Measured outputs identical",
        "yes" if result.outputs_are_identical else "no",
    )

    peak_memory = result.peak_gpu_memory_used_mib
    total_memory = result.total_gpu_memory_mib

    if peak_memory is not None and total_memory is not None:
        table.add_row(
            "Peak GPU memory",
            (
                f"{peak_memory} / {total_memory} MiB "
                f"({total_memory - peak_memory} MiB free)"
            ),
        )

    average_utilization = result.average_gpu_utilization_percent
    peak_utilization = result.peak_gpu_utilization_percent

    if average_utilization is not None and peak_utilization is not None:
        table.add_row(
            "GPU utilization",
            (f"{average_utilization:.1f}% average, {peak_utilization}% peak"),
        )

    console.print()
    console.print(table)

    for gpu_error in result.gpu_errors:
        console.print(
            f"GPU monitoring warning: {gpu_error}",
            style="bold yellow",
            markup=False,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for the benchmark command."""

    parser = argparse.ArgumentParser(
        prog="home-llm-benchmark",
        description=("Measure one installed local Ollama model using a fixed prompt."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=local_model_name,
        required=True,
        help="Installed local Ollama model to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=positive_integer,
        default=DEFAULT_MEASURED_RUNS,
        help="Number of measured runs after one warm-up.",
    )
    parser.add_argument(
        "--context-length",
        type=positive_integer,
        default=DEFAULT_CONTEXT_LENGTH,
        help="Context length used by every generation.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=positive_integer,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens for every generation.",
    )
    parser.add_argument(
        "--gpu-report",
        action="store_true",
        help="Collect GPU memory and utilization samples.",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=positive_float,
        default=DEFAULT_GPU_SAMPLE_INTERVAL,
        help="Seconds between GPU samples.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark command."""

    arguments = build_parser().parse_args(argv)
    console = Console()

    backend = OllamaBackend(
        model=arguments.model,
        context_length=arguments.context_length,
        max_output_tokens=arguments.max_output_tokens,
        temperature=BENCHMARK_TEMPERATURE,
        seed=BENCHMARK_SEED,
    )

    gpu_monitor_factory: GpuMonitorFactory | None = None

    if arguments.gpu_report:
        gpu_monitor_factory = partial(
            NvidiaSmiGpuMonitor,
            interval_seconds=arguments.gpu_sample_interval,
        )

    console.print(
        (
            f"Benchmarking {arguments.model}: "
            f"1 warm-up + {arguments.runs} measured runs."
        ),
        markup=False,
    )

    try:
        result = run_benchmark(
            backend,
            model_name=arguments.model,
            measured_runs=arguments.runs,
            gpu_monitor_factory=gpu_monitor_factory,
        )
    except (RuntimeError, ValueError) as error:
        console.print(
            f"Benchmark failed: {error}",
            style="bold red",
            markup=False,
        )
        return 1

    render_benchmark_result(result, console)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
