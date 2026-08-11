"""
Intelligent engine routing example.

This example demonstrates how to use the framework's intelligent
routing capabilities to automatically select between PaddleOCR and TrOCR
based on confidence scores.
"""

from pathlib import Path

from ocr_framework import create_paddle_pipeline


def main() -> None:
    """Run OCR with intelligent routing and fallback."""
    # Create a PaddleOCR pipeline (routing will be automatic)
    pipeline = create_paddle_pipeline(language="en", use_gpu=False)

    # Process an image with routing enabled
    image_path = Path("samples/sample_image.png")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        print("Please place a sample image at samples/sample_image.png")
        return

    try:
        # Use the new routing-enabled method
        result = pipeline.run_with_routing(image_path)

        # Print the results
        print(f"Processed {result.page_count} page(s)")
        print()

        for page_result in result.pages:
            print(f"Page {page_result.page_index}:")
            print(f"  Engine: {page_result.engine_used}")
            print(f"  Aggregate confidence: {page_result.aggregate_confidence:.2f}")
            print(f"  Lines detected: {len(page_result.lines)}")
            print(f"  Routing decisions: {len(page_result.routing_decisions)}")
            print()

            for i, line in enumerate(page_result.lines, 1):
                print(f"  Line {i}: {line.text}")
                print(f"    Confidence: {line.confidence:.2f}")
            print()

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
