"""Run unittest discovery with optional per-test GPU reports."""

import argparse
import unittest
from collections.abc import Sequence
from unittest.runner import _WritelnDecorator

from home_llm.gpu_monitor import (
    NvidiaSmiGpuMonitor,
)

DEFAULT_TEST_DIRECTORY = "tests"
DEFAULT_TEST_PATTERN = "test_*.py"
DEFAULT_GPU_SAMPLE_INTERVAL = 0.1


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


class GpuReportingTestResult(unittest.TextTestResult):
    """Collect and print one GPU report for every test."""

    def __init__(
        self,
        stream: _WritelnDecorator,
        descriptions: bool,
        verbosity: int,
        *,
        gpu_sample_interval: float,
    ) -> None:
        super().__init__(stream, descriptions, verbosity)
        self._gpu_sample_interval = gpu_sample_interval
        self._gpu_monitor: NvidiaSmiGpuMonitor | None = None

    def startTest(self, test: unittest.case.TestCase) -> None:
        """Start GPU sampling before the test body executes."""

        super().startTest(test)

        self._gpu_monitor = NvidiaSmiGpuMonitor(
            interval_seconds=self._gpu_sample_interval,
        )
        self._gpu_monitor.start()

    def stopTest(self, test: unittest.case.TestCase) -> None:
        """Stop sampling and print the report after the test status."""

        gpu_monitor = self._gpu_monitor
        self._gpu_monitor = None

        super().stopTest(test)

        if gpu_monitor is None:
            self.stream.writeln(" GPU monitoring unavailable: monitor was not started.")
            return

        report = gpu_monitor.stop()
        self.stream.writeln(f" {report.summary()}")
        self.stream.flush()


class GpuReportingTestRunner(unittest.TextTestRunner):
    """Create test results configured for GPU sampling."""

    def __init__(
        self,
        *,
        gpu_sample_interval: float,
        verbosity: int = 2,
    ) -> None:
        super().__init__(verbosity=verbosity)
        self._gpu_sample_interval = gpu_sample_interval

    def _makeResult(self) -> GpuReportingTestResult:
        return GpuReportingTestResult(
            self.stream,
            self.descriptions,
            self.verbosity,
            gpu_sample_interval=self._gpu_sample_interval,
        )


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for test discovery and GPU reporting."""

    parser = argparse.ArgumentParser(
        prog="home-llm-tests",
        description=("Run HomeLLM unittest discovery with optional GPU reports."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--gpu-report",
        action="store_true",
        help=("Monitor the GPU during every test and print a report after its result."),
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=positive_float,
        default=DEFAULT_GPU_SAMPLE_INTERVAL,
        help="Seconds between GPU measurements.",
    )
    parser.add_argument(
        "--start-directory",
        default=DEFAULT_TEST_DIRECTORY,
        help="Directory where unittest discovery starts.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_TEST_PATTERN,
        help="Filename pattern used during unittest discovery.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Discover tests, run them, and return a process exit code."""

    arguments = build_parser().parse_args(argv)

    suite = unittest.defaultTestLoader.discover(
        start_dir=arguments.start_directory,
        pattern=arguments.pattern,
    )

    if arguments.gpu_report:
        runner: unittest.TextTestRunner = GpuReportingTestRunner(
            gpu_sample_interval=arguments.gpu_sample_interval,
        )
    else:
        runner = unittest.TextTestRunner(verbosity=2)

    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
