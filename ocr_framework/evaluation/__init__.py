"""OCR evaluation and benchmarking framework."""

from ocr_framework.evaluation.benchmark import Benchmark, BenchmarkReport, BenchmarkResult
from ocr_framework.evaluation.cer import CERCalculator
from ocr_framework.evaluation.dataset_loader import DatasetLoader, EvaluationSample
from ocr_framework.evaluation.wer import WERCalculator

__all__ = [
    "Benchmark",
    "BenchmarkReport",
    "BenchmarkResult",
    "CERCalculator",
    "DatasetLoader",
    "EvaluationSample",
    "WERCalculator",
]
