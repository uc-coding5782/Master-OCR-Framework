"""
OCR evaluation example.

This example demonstrates how to evaluate OCR performance using
CER, WER, and benchmarking metrics.
"""

from pathlib import Path

from ocr_framework import create_paddle_pipeline
from ocr_framework.evaluation import (
    Benchmark,
    CERCalculator,
    DatasetLoader,
    WERCalculator,
)


def main() -> None:
    """Run OCR evaluation on a test dataset."""
    # Initialize calculators
    cer_calc = CERCalculator()
    wer_calc = WERCalculator()

    # Sample evaluation data (in practice, load from dataset)
    ground_truths = [
        "Hello world",
        "This is a test",
        "OCR evaluation framework",
    ]

    hypotheses = [
        "Hello world",  # Perfect match
        "This is test",  # Missing 'a'
        "OCR evalution framework",  # Typo
    ]

    # Calculate CER
    print("Character Error Rate (CER):")
    for gt, hyp in zip(ground_truths, hypotheses):
        cer = cer_calc.calculate(gt, hyp)
        print(f"  '{gt}' vs '{hyp}': {cer:.3f}")

    cer_stats = cer_calc.calculate_batch(ground_truths, hypotheses)
    print(f"\nCER Statistics: {cer_stats}")

    # Calculate WER
    print("\nWord Error Rate (WER):")
    for gt, hyp in zip(ground_truths, hypotheses):
        wer = wer_calc.calculate(gt, hyp)
        print(f"  '{gt}' vs '{hyp}': {wer:.3f}")

    wer_stats = wer_calc.calculate_batch(ground_truths, hypotheses)
    print(f"\nWER Statistics: {wer_stats}")

    # Benchmark pipeline performance
    print("\n" + "=" * 50)
    print("BENCHMARKING")
    print("=" * 50)

    try:
        pipeline = create_paddle_pipeline(language="en", use_gpu=False)

        benchmark = Benchmark()

        def run_ocr() -> str:
            """Dummy OCR function for benchmarking."""
            # In practice, this would run actual OCR
            import time
            time.sleep(0.1)  # Simulate processing
            return "dummy result"

        result = benchmark.run("paddle_ocr", run_ocr, iterations=5)

        print(f"Benchmark: {result.name}")
        print(f"  Duration: {result.duration_seconds:.3f}s")
        print(f"  Memory: {result.memory_mb:.2f} MB")
        print(f"  Success: {result.success}")

    except ImportError:
        print("PaddleOCR not installed, skipping benchmark")

    # Dataset loading example
    print("\n" + "=" * 50)
    print("DATASET LOADING")
    print("=" * 50)

    loader = DatasetLoader()

    # Example: Load from JSON (commented out as we don't have a real dataset)
    # samples = loader.load_from_json(Path("dataset.json"))
    # print(f"Loaded {len(samples)} samples from JSON")

    # Example: Load from paired directories
    # samples = loader.load_from_directory(
    #     Path("images"),
    #     Path("texts")
    # )
    # print(f"Loaded {len(samples)} samples from directories")

    print("Dataset loading requires actual dataset files.")
    print("Supported formats: JSON, paired directories, text files")


if __name__ == "__main__":
    main()
