"""Tests for evaluation framework components."""

from pathlib import Path

import pytest

from ocr_framework.evaluation.benchmark import Benchmark, BenchmarkReport, BenchmarkResult
from ocr_framework.evaluation.cer import CERCalculator
from ocr_framework.evaluation.dataset_loader import DatasetLoader, EvaluationSample
from ocr_framework.evaluation.wer import WERCalculator


class TestCERCalculator:
    """Tests for CERCalculator."""

    def test_perfect_match(self) -> None:
        """Test CER calculation for perfect match."""
        calc = CERCalculator()
        cer = calc.calculate("hello world", "hello world")
        assert cer == 0.0

    def test_substitution(self) -> None:
        """Test CER calculation with substitution."""
        calc = CERCalculator()
        cer = calc.calculate("hello world", "helo world")
        assert cer == 1/11  # 1 char out of 11 (including space)

    def test_insertion(self) -> None:
        """Test CER calculation with insertion."""
        calc = CERCalculator()
        cer = calc.calculate("hello", "helllo")
        assert cer == 0.2  # 1 insertion out of 5

    def test_deletion(self) -> None:
        """Test CER calculation with deletion."""
        calc = CERCalculator()
        cer = calc.calculate("hello", "helo")
        assert cer == 0.2  # 1 deletion out of 5

    def test_empty_ground_truth(self) -> None:
        """Test CER calculation with empty ground truth."""
        calc = CERCalculator()
        cer = calc.calculate("", "hello")
        assert cer == 1.0

    def test_both_empty(self) -> None:
        """Test CER calculation with both strings empty."""
        calc = CERCalculator()
        cer = calc.calculate("", "")
        assert cer == 0.0

    def test_batch_calculation(self) -> None:
        """Test batch CER calculation."""
        calc = CERCalculator()
        ground_truths = ["hello", "world", "test"]
        hypotheses = ["helo", "world", "tost"]

        stats = calc.calculate_batch(ground_truths, hypotheses)

        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert 0.0 <= stats["mean"] <= 1.0

    def test_batch_calculation_mismatched_lengths(self) -> None:
        """Test batch CER calculation with mismatched lengths."""
        calc = CERCalculator()
        ground_truths = ["hello", "world"]
        hypotheses = ["helo"]

        with pytest.raises(ValueError, match="same length"):
            calc.calculate_batch(ground_truths, hypotheses)


class TestWERCalculator:
    """Tests for WERCalculator."""

    def test_perfect_match(self) -> None:
        """Test WER calculation for perfect match."""
        calc = WERCalculator()
        wer = calc.calculate("hello world", "hello world")
        assert wer == 0.0

    def test_substitution(self) -> None:
        """Test WER calculation with substitution."""
        calc = WERCalculator()
        wer = calc.calculate("hello world", "helo world")
        assert wer == 0.5  # 1 word out of 2

    def test_insertion(self) -> None:
        """Test WER calculation with insertion."""
        calc = WERCalculator()
        wer = calc.calculate("hello", "hello world")
        assert wer == 1.0  # 1 insertion out of 1

    def test_deletion(self) -> None:
        """Test WER calculation with deletion."""
        calc = WERCalculator()
        wer = calc.calculate("hello world", "hello")
        assert wer == 0.5  # 1 deletion out of 2

    def test_empty_ground_truth(self) -> None:
        """Test WER calculation with empty ground truth."""
        calc = WERCalculator()
        wer = calc.calculate("", "hello world")
        assert wer == 1.0

    def test_both_empty(self) -> None:
        """Test WER calculation with both strings empty."""
        calc = WERCalculator()
        wer = calc.calculate("", "")
        assert wer == 0.0

    def test_batch_calculation(self) -> None:
        """Test batch WER calculation."""
        calc = WERCalculator()
        ground_truths = ["hello world", "test case", "foo bar"]
        hypotheses = ["helo world", "test case", "foo"]

        stats = calc.calculate_batch(ground_truths, hypotheses)

        assert "mean" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" in stats
        assert 0.0 <= stats["mean"] <= 1.0

    def test_batch_calculation_mismatched_lengths(self) -> None:
        """Test batch WER calculation with mismatched lengths."""
        calc = WERCalculator()
        ground_truths = ["hello world", "test case"]
        hypotheses = ["helo"]

        with pytest.raises(ValueError, match="same length"):
            calc.calculate_batch(ground_truths, hypotheses)


class TestBenchmark:
    """Tests for Benchmark system."""

    def test_single_benchmark(self) -> None:
        """Test running a single benchmark."""
        benchmark = Benchmark()

        def dummy_func() -> str:
            return "result"

        result = benchmark.run("test_benchmark", dummy_func)

        assert result.name == "test_benchmark"
        assert result.success is True
        assert result.duration_seconds >= 0.0
        assert result.memory_mb >= 0.0

    def test_benchmark_with_error(self) -> None:
        """Test benchmark with function that raises error."""
        benchmark = Benchmark()

        def failing_func() -> str:
            raise ValueError("Test error")

        result = benchmark.run("failing_benchmark", failing_func)

        assert result.name == "failing_benchmark"
        assert result.success is False
        assert result.error is not None

    def test_benchmark_with_iterations(self) -> None:
        """Test benchmark with multiple iterations."""
        benchmark = Benchmark()

        def dummy_func() -> str:
            return "result"

        result = benchmark.run("test_benchmark", dummy_func, iterations=5)

        assert result.metadata["iterations"] == 5
        assert result.success is True

    def test_batch_benchmark(self) -> None:
        """Test running multiple benchmarks."""
        benchmark = Benchmark()

        benchmarks = [
            ("bench1", lambda: "result1", 1),
            ("bench2", lambda: "result2", 1),
        ]

        report = benchmark.run_batch(benchmarks)

        assert len(report.results) == 2
        assert report.total_duration >= 0.0
        assert report.average_duration >= 0.0
        assert report.success_rate == 1.0

    def test_clear_results(self) -> None:
        """Test clearing benchmark results."""
        benchmark = Benchmark()

        benchmark.run("test", lambda: "result")
        assert len(benchmark._results) == 1

        benchmark.clear()
        assert len(benchmark._results) == 0

    def test_get_report(self) -> None:
        """Test getting benchmark report."""
        benchmark = Benchmark()

        benchmark.run("test", lambda: "result")
        report = benchmark.get_report()

        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 1


class TestDatasetLoader:
    """Tests for DatasetLoader."""

    def test_load_from_json_file_not_found(self) -> None:
        """Test loading from non-existent JSON file."""
        loader = DatasetLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_json(Path("/nonexistent/file.json"))

    def test_load_from_json(self, tmp_path: Path) -> None:
        """Test loading dataset from JSON file."""
        import json

        # Create test JSON file
        data = [
            {"image": "test1.jpg", "text": "Sample text 1", "metadata": {"key": "value"}},
            {"image": "test2.png", "text": "Sample text 2", "metadata": {}},
        ]

        json_path = tmp_path / "dataset.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loader = DatasetLoader()
        samples = loader.load_from_json(json_path)

        assert len(samples) == 2
        assert samples[0].ground_truth == "Sample text 1"
        assert samples[1].ground_truth == "Sample text 2"

    def test_load_from_directory_not_found(self) -> None:
        """Test loading from non-existent directory."""
        loader = DatasetLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_directory(Path("/nonexistent/images"), Path("/nonexistent/texts"))

    def test_load_from_directory(self, tmp_path: Path) -> None:
        """Test loading dataset from paired directories."""
        # Create images directory
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # Create texts directory
        texts_dir = tmp_path / "texts"
        texts_dir.mkdir()

        # Create sample files
        (images_dir / "test1.jpg").touch()
        (images_dir / "test2.png").touch()

        (texts_dir / "test1.txt").write_text("Text 1", encoding="utf-8")
        (texts_dir / "test2.txt").write_text("Text 2", encoding="utf-8")

        loader = DatasetLoader()
        samples = loader.load_from_directory(images_dir, texts_dir)

        assert len(samples) == 2
        assert samples[0].ground_truth == "Text 1"
        assert samples[1].ground_truth == "Text 2"

    def test_load_from_text_file_not_found(self) -> None:
        """Test loading from non-existent text file."""
        loader = DatasetLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_from_text_file(Path("/nonexistent/file.txt"))

    def test_load_from_text_file(self, tmp_path: Path) -> None:
        """Test loading dataset from text file."""
        # Create test text file
        text_file = tmp_path / "dataset.txt"
        text_file.write_text(
            "test1.jpg|Sample text 1\ntest2.png|Sample text 2\n# Comment line\n",
            encoding="utf-8",
        )

        loader = DatasetLoader()
        samples = loader.load_from_text_file(text_file)

        assert len(samples) == 2
        assert samples[0].ground_truth == "Sample text 1"
        assert samples[1].ground_truth == "Sample text 2"

    def test_get_ground_truths(self) -> None:
        """Test extracting ground truths from samples."""
        samples = [
            EvaluationSample(Path("img1.jpg"), "text1", {}),
            EvaluationSample(Path("img2.jpg"), "text2", {}),
        ]

        loader = DatasetLoader()
        truths = loader.get_ground_truths(samples)

        assert truths == ["text1", "text2"]

    def test_get_image_paths(self) -> None:
        """Test extracting image paths from samples."""
        samples = [
            EvaluationSample(Path("img1.jpg"), "text1", {}),
            EvaluationSample(Path("img2.jpg"), "text2", {}),
        ]

        loader = DatasetLoader()
        paths = loader.get_image_paths(samples)

        assert len(paths) == 2
        assert paths[0] == Path("img1.jpg")
        assert paths[1] == Path("img2.jpg")
