"""
TrOCR handwriting recognition example.

This example demonstrates how to use Microsoft TrOCR for
handwritten text recognition.
"""

from pathlib import Path

from ocr_framework import create_trocr_pipeline


def main() -> None:
    """Run TrOCR handwriting recognition on a sample image."""
    # Create a TrOCR pipeline
    pipeline = create_trocr_pipeline(
        model_name="microsoft/trocr-base-handwritten",
        use_gpu=False,
    )

    # Process an image
    image_path = Path("samples/handwriting.png")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please place a handwriting sample at samples/handwriting.png")
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

    except ImportError as e:
        print(f"Error: {e}")
        print("To use TrOCR, install transformers and torch:")
        print("pip install transformers torch")
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
