"""Common benchmark runner: identical samples, identical crops, identical
metrics for every adapter."""

from .runner import FER7, expected_calibration_error, run_benchmark

__all__ = ["run_benchmark", "FER7", "expected_calibration_error"]
