"""Deterministic tests for GPU monitoring and reporting."""

import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from home_llm.gpu_monitor import (
    GpuReport,
    GpuSample,
    NvidiaSmiGpuMonitor,
    read_nvidia_smi,
)


class GpuReportTests(unittest.TestCase):
    def test_summarizes_gpu_samples(self) -> None:
        report = GpuReport(
            duration_seconds=0.6,
            samples=(
                GpuSample(
                    elapsed_seconds=0.0,
                    memory_used_mib=7000,
                    memory_total_mib=12227,
                    utilization_percent=10,
                ),
                GpuSample(
                    elapsed_seconds=0.5,
                    memory_used_mib=7200,
                    memory_total_mib=12227,
                    utilization_percent=90,
                ),
            ),
        )

        self.assertEqual(report.minimum_memory_used_mib, 7000)
        self.assertEqual(report.peak_memory_used_mib, 7200)
        self.assertEqual(report.average_utilization_percent, 50.0)
        self.assertEqual(report.peak_utilization_percent, 90)
        self.assertEqual(
            report.summary(),
            (
                "GPU: 2 samples over 0.60 s"
                " | memory 7000-7200 MiB / 12227 MiB"
                " | peak headroom 5027 MiB"
                " | utilization average 50.0%, peak 90%"
            ),
        )

    def test_reports_missing_samples(self) -> None:
        report = GpuReport(
            duration_seconds=0.1,
            samples=(),
            error="simulated measurement failure",
        )

        self.assertEqual(
            report.summary(),
            ("GPU monitoring unavailable: simulated measurement failure"),
        )


class NvidiaSmiGpuMonitorTests(unittest.TestCase):
    def test_collects_injected_baseline_reading(self) -> None:
        monitor = NvidiaSmiGpuMonitor(
            interval_seconds=60.0,
            reading_function=lambda: (7100, 12227, 25),
        )

        monitor.start()
        report = monitor.stop()

        self.assertEqual(len(report.samples), 1)
        self.assertEqual(report.samples[0].memory_used_mib, 7100)
        self.assertEqual(report.samples[0].memory_total_mib, 12227)
        self.assertEqual(report.samples[0].utilization_percent, 25)
        self.assertIsNone(report.error)

    def test_rejects_invalid_sampling_interval(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "sampling interval must be greater than zero",
        ):
            NvidiaSmiGpuMonitor(interval_seconds=0)

    @patch("home_llm.gpu_monitor.subprocess.run")
    def test_executes_and_parses_nvidia_smi_query(
        self,
        run_mock,
    ) -> None:
        run_mock.return_value = CompletedProcess(
            args=[],
            returncode=0,
            stdout="7200, 12227, 97\n",
            stderr="",
        )

        reading = read_nvidia_smi(gpu_index=0)

        self.assertEqual(reading, (7200, 12227, 97))
        run_mock.assert_called_once_with(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )


if __name__ == "__main__":
    unittest.main()
