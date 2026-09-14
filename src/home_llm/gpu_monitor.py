"""Reusable NVIDIA GPU sampling for tests and benchmarks."""

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from threading import Event, Lock, Thread
from typing import Protocol

GpuReading = tuple[int, int, int]
GpuReadingFunction = Callable[[], GpuReading]


class NvidiaSmiError(RuntimeError):
    """Raised when an NVIDIA GPU measurement cannot be collected."""


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One GPU measurement captured during an operation."""

    elapsed_seconds: float
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int


@dataclass(frozen=True, slots=True)
class GpuReport:
    """Summary of GPU sampples captured during one operation."""

    duration_seconds: float
    # Tuple containing many items of this Type.
    # Useful when returning a completed set of measurements that should stay fixed.
    # In contrast the list can be modified: ie add, clear.
    samples: tuple[GpuSample, ...]
    error: str | None = None

    @property
    def minimum_memory_used_mib(self) -> int | None:
        if not self.samples:
            return None

        return min(sample.memory_used_mib for sample in self.samples)

    @property
    def peak_memory_used_mib(self) -> int | None:
        if not self.samples:
            return None

        return max(sample.memory_used_mib for sample in self.samples)

    @property
    def peak_utilization_percent(self) -> int | None:
        if not self.samples:
            return None

        return max(sample.utilization_percent for sample in self.samples)

    @property
    def average_utilization_percent(self) -> float | None:
        if not self.samples:
            return None

        return fmean(sample.utilization_percent for sample in self.samples)

    def summary(self) -> str:
        """Return a compact human-readable report."""

        if not self.samples:
            reason = self.error or "no samples were collected"
            return f"GPU monitoring unavailable: {reason}"

        minimum_memory = self.minimum_memory_used_mib
        peak_memory = self.peak_memory_used_mib
        average_utilization = self.average_utilization_percent
        peak_utilization = self.peak_utilization_percent
        total_memory = self.samples[0].memory_total_mib

        assert minimum_memory is not None
        assert peak_memory is not None
        assert average_utilization is not None
        assert peak_utilization is not None

        peak_headroom = max(total_memory - peak_memory, 0)

        summary = (
            f"GPU: {len(self.samples)} samples over "
            f"{self.duration_seconds:.2f} s"
            f" | memory {minimum_memory}-{peak_memory} MiB"
            f" / {total_memory} MiB"
            f" | peak headroom {peak_headroom} MiB"
            f" | utilization average "
            f"{average_utilization:.1f}%, peak "
            f"{peak_utilization}%"
        )

        if self.error is not None:
            summary += f"| warning: {self.error}"

        return summary


# TODO Why do we need this ?
class GpuMonitor(Protocol):
    """Interface implemented by GPU monitoring backends."""

    def start(self) -> None:
        """Begin collectiong GPU measurements."""

    def stop(self) -> GpuReport:
        """Stop collection and return the resulting report."""


def read_nvidia_smi(gpu_index: int = 0) -> GpuReading:
    """Read memory and utilization for one NVIDIA GPU.

    Returns:
        A tuple containing used memory in MiB, total memory in MiB,
        and GPU utilization as a percentage.
    """

    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        completed_process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
    except FileNotFoundError as error:
        raise NvidiaSmiError(
            "nvidia-smi was not found in the  Ubuntu/WSL environment."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise NvidiaSmiError("nvidia-smi did not respond within 5 seconds.") from error
    except OSError as error:
        raise NvidiaSmiError(f"nvidia-smi could not be executed: {error}") from error

    if completed_process.returncode != 0:
        error_text = (
            completed_process.stderr.strip() or completed_process.stdout.strip()
        )
        raise NvidiaSmiError(
            "nvidia-smi returned a nonzero exit status"
            + (f": {error_text}" if error_text else ".")
        )

    output_lines = completed_process.stdout.strip().splitlines()

    if len(output_lines) != 1:
        raise NvidiaSmiError(
            "Expected one GPU measurement from nvidia-smi, "
            f"received {len(output_lines)}."
        )

    fields = [field.strip() for field in output_lines[0].split(",")]

    if len(fields) != 3:
        raise NvidiaSmiError(
            "Expected memory used, memory total, and utilization "
            f"from nvidia-smi; received {output_lines[0]!r}."
        )

    try:
        memory_used_mib = int(fields[0])
        memory_total_mib = int(fields[1])
        utilization_percent = int(fields[2])
    except ValueError as error:
        raise NvidiaSmiError(
            f"nvidia-smi returned a nonnumeric GPU measurement: {output_lines[0]!r}"
        ) from error

    return (
        memory_used_mib,
        memory_total_mib,
        utilization_percent,
    )


class NvidiaSmiGpuMonitor:
    """Sample one NVIDIA GPU in a background thread."""

    def __init__(
        self,
        *,
        gpu_index: int = 0,
        interval_seconds: float = 0.1,
        reading_function: GpuReadingFunction | None = None,
    ) -> None:
        if gpu_index < 0:
            raise ValueError("GPU index must not be negative.")

        if interval_seconds <= 0:
            raise ValueError("GPU sampling interval must be greater than zero.")

        self._gpu_index = gpu_index
        self._interval_second = interval_seconds
        self._reading_function = (
            reading_function
            if reading_function is not None
            else lambda: read_nvidia_smi(self._gpu_index)
        )

        self._stop_event = Event()
        self._lock = Lock()
        self._samples: list[GpuSample] = []
        self._error: str | None = None
        self._started_ns: int | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        """Capture a baseliune measurement and start background sampling."""

        if self._started_ns is not None:
            raise RuntimeError("This GPU monitor has already been started.")

        self._started_ns = time.monotonic_ns()
        self._record_sample()

        if self._error is not None:
            return

        self._thread = Thread(
            target=self._sampling_loop, name="home-llm-gpu-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> GpuReport:
        """Stop backgrounds sampling and return an immutable report."""

        if self._started_ns is None:
            raise RuntimeError("The GPU monitor has not been started.")

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=6.0)

            if self._thread.is_alive() and self._error is None:
                self._error = "GPU sampling thread did not stop within 6 seconds."

        duration_seconds = (time.monotonic_ns() - self._started_ns) / 1_000_000_000

        with self._lock:
            samples = tuple(self._samples)

        return GpuReport(
            duration_seconds=duration_seconds,
            samples=samples,
            error=self._error,
        )

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self._interval_second):
            self._record_sample()

            if self._error is not None:
                return

    def _record_sample(self) -> None:
        assert self._started_ns is not None

        try:
            (
                memory_used_mib,
                memory_total_mib,
                utilization_percent,
            ) = self._reading_function()
        except NvidiaSmiError as error:
            self._error = str(error)
            return

        elapsed_seconds = (time.monotonic_ns() - self._started_ns) / 1_000_000_000

        sample = GpuSample(
            elapsed_seconds=elapsed_seconds,
            memory_used_mib=memory_used_mib,
            memory_total_mib=memory_total_mib,
            utilization_percent=utilization_percent,
        )

        with self._lock:
            self._samples.append(sample)
