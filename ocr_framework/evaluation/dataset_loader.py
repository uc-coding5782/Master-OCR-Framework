"""Dataset loader for OCR evaluation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationSample:
    """A single evaluation sample with ground truth."""

    image_path: Path
    ground_truth: str
    metadata: dict


class DatasetLoader:
    """Load evaluation datasets for OCR performance assessment.

    The DatasetLoader supports multiple dataset formats including
    JSON files with ground truth annotations and directory-based
    datasets with text files.
    """

    def load_from_json(self, json_path: Path) -> list[EvaluationSample]:
        """Load evaluation dataset from a JSON file.

        Args:
            json_path: Path to JSON file with dataset.

        Returns:
            List of EvaluationSample objects.

        Expected JSON format:
        [
            {
                "image": "path/to/image1.jpg",
                "text": "Ground truth text",
                "metadata": {}
            },
            ...
        ]
        """
        if not json_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        base_dir = json_path.parent

        for item in data:
            image_path = base_dir / item["image"]
            ground_truth = item["text"]
            metadata = item.get("metadata", {})

            samples.append(
                EvaluationSample(
                    image_path=image_path,
                    ground_truth=ground_truth,
                    metadata=metadata,
                )
            )

        return samples

    def load_from_directory(
        self,
        images_dir: Path,
        texts_dir: Path,
    ) -> list[EvaluationSample]:
        """Load evaluation dataset from paired image and text directories.

        Args:
            images_dir: Directory containing images.
            texts_dir: Directory containing corresponding text files.

        Returns:
            List of EvaluationSample objects.

        The text files should have the same base name as their corresponding
        images, with .txt extension.
        """
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        if not texts_dir.exists():
            raise FileNotFoundError(f"Texts directory not found: {texts_dir}")

        samples = []
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
                continue

            # Look for corresponding text file
            text_path = texts_dir / f"{image_path.stem}.txt"

            if not text_path.exists():
                continue

            # Read ground truth
            with open(text_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()

            samples.append(
                EvaluationSample(
                    image_path=image_path,
                    ground_truth=ground_truth,
                    metadata={"text_path": str(text_path)},
                )
            )

        return samples

    def load_from_text_file(self, text_file: Path) -> list[EvaluationSample]:
        """Load evaluation dataset from a simple text file.

        Args:
            text_file: Path to text file with dataset.

        Returns:
            List of EvaluationSample objects.

        Expected format (one sample per line):
        image_path|ground_truth text
        """
        if not text_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {text_file}")

        samples = []
        base_dir = text_file.parent

        with open(text_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("|", 1)
                if len(parts) != 2:
                    continue

                image_path = base_dir / parts[0].strip()
                ground_truth = parts[1].strip()

                samples.append(
                    EvaluationSample(
                        image_path=image_path,
                        ground_truth=ground_truth,
                        metadata={},
                    )
                )

        return samples

    def get_ground_truths(self, samples: Sequence[EvaluationSample]) -> list[str]:
        """Extract ground truth texts from samples.

        Args:
            samples: List of evaluation samples.

        Returns:
            List of ground truth texts.
        """
        return [sample.ground_truth for sample in samples]

    def get_image_paths(self, samples: Sequence[EvaluationSample]) -> list[Path]:
        """Extract image paths from samples.

        Args:
            samples: List of evaluation samples.

        Returns:
            List of image paths.
        """
        return [sample.image_path for sample in samples]
