"""
Batch processing example.

This example demonstrates how to process an entire folder of images
using the framework's batch processing capabilities.
"""

from pathlib import Path

from ocr_framework import create_batch_processor, create_paddle_pipeline


def main() -> None:
    """Run batch OCR on a folder of images."""
    # Create a PaddleOCR pipeline
    pipeline = create_paddle_pipeline(language="en", use_gpu=False)

    # Create a batch processor
    input_dir = Path("samples")
    output_dir = Path("outputs")

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print("Please create a 'samples' directory with some images")
        return

    try:
        processor = create_batch_processor(
            pipeline=pipeline,
            output_dir=str(output_dir),
            silent=False,  # Set to True to suppress progress output
        )

        # Process all images in the directory
        report = processor.process_directory(input_dir)

        # Print summary
        print("\n" + "=" * 50)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total files: {report.total}")
        print(f"Succeeded: {report.succeeded}")
        print(f"Failed: {report.failed}")
        print(f"Output directory: {output_dir}")

        # Show details of failed jobs
        if report.failed > 0:
            print("\nFailed files:")
            for job in report.jobs:
                if job.status.value == "failed":
                    print(f"  - {job.input_path.name}: {job.error}")

    except Exception as e:
        print(f"Error during batch processing: {e}")


if __name__ == "__main__":
    main()
