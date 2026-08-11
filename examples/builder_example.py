"""
Pipeline builder example.

This example demonstrates how to use the PipelineBuilder to create
customized OCR pipelines with specific configurations.
"""

from pathlib import Path

from ocr_framework import PipelineBuilder
from ocr_framework.postprocessing.filters.confidence_filter import ConfidenceFilter
from ocr_framework.preprocessing.steps.deskew import DeskewStep
from ocr_framework.preprocessing.steps.contrast import ContrastStep


def main() -> None:
    """Run OCR using the pipeline builder."""
    # Create a custom pipeline using the builder
    builder = PipelineBuilder()

    # Configure with a profile
    builder.with_profile("document")
    builder.with_language("en")

    # Wire up PaddleOCR components
    builder.with_paddle_ocr()

    # Build the pipeline
    pipeline = builder.build()

    # Process an image
    image_path = Path("samples/sample_image.png")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please place a sample image at samples/sample_image.png")
        return

    try:
        # Run the OCR pipeline
        result = pipeline.run(image_path)

        # Print the results
        print(f"Processed {result.page_count} page(s)")
        print()

        for page_result in result.pages:
            print(f"Page {page_result.page_index}:")
            print(f"  Engine: {page_result.engine_used}")
            print(f"  Aggregate confidence: {page_result.aggregate_confidence:.2f}")
            print(f"  Lines detected: {len(page_result.lines)}")
            print()

            for i, line in enumerate(page_result.lines, 1):
                print(f"  Line {i}: {line.text}")
                print(f"    Confidence: {line.confidence:.2f}")
            print()

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
