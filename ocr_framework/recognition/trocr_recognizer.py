"""Microsoft TrOCR recognition engine implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from transformers import TrOCRProcessor, TrOCRForCausalLM
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

from ocr_framework.exceptions import RecognitionError
from ocr_framework.models.bbox import BoundingBox, Point, Polygon
from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion, OCRLine
from ocr_framework.recognition.base import TextRecognizer

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class TrOCRRecognizer(TextRecognizer):
    """Microsoft TrOCR text recognizer for handwritten and printed text.

    TrOCR is a transformer-based OCR model that excels at handwriting
    recognition while also supporting printed text. It uses an
    encoder-decoder architecture similar to machine translation models.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
        use_gpu: bool = False,
    ) -> None:
        """Initialize the TrOCR recognizer.

        Args:
            model_name: HuggingFace model name. Options include:
                - 'microsoft/trocr-base-handwritten' (default)
                - 'microsoft/trocr-small-handwritten'
                - 'microsoft/trocr-base-printed'
            use_gpu: Whether to use GPU acceleration.

        Raises:
            ImportError: If transformers library is not installed.
        """
        if not TROCR_AVAILABLE:
            raise ImportError(
                "transformers library is not installed. Install it with: "
                "pip install transformers torch"
            )

        self._model_name = model_name
        self._use_gpu = use_gpu
        self._processor: TrOCRProcessor | None = None
        self._model: TrOCRForCausalLM | None = None

    @property
    def name(self) -> str:
        """Return the recognizer identifier."""
        return "trocr_recognizer"

    def _get_model(self) -> tuple[TrOCRProcessor, TrOCRForCausalLM]:
        """Lazy-load the TrOCR model and processor.

        Returns:
            Tuple of (processor, model).

        Raises:
            RecognitionError: If model loading fails.
        """
        if self._processor is None or self._model is None:
            try:
                device = "cuda" if self._use_gpu else "cpu"
                self._processor = TrOCRProcessor.from_pretrained(self._model_name)
                self._model = TrOCRForCausalLM.from_pretrained(self._model_name).to(device)
            except Exception as exc:
                raise RecognitionError(f"Failed to load TrOCR model: {exc}") from exc

        return self._processor, self._model

    def recognize(
        self,
        image: ImagePayload,
        regions: list[DetectionRegion],
        context: PipelineContext,
    ) -> list[OCRLine]:
        """Recognize text from detected regions.

        Args:
            image: Input image payload.
            regions: Detection regions to transcribe.
            context: Pipeline execution context.

        Returns:
            Recognized OCR lines.

        Raises:
            RecognitionError: If recognition fails.
        """
        try:
            processor, model = self._get_model()

            # Convert image to PIL format
            pil_image = self._convert_to_pil(image)

            # TrOCR processes the entire image at once (no region-based processing)
            # For region-based processing, we would need to crop each region
            # For now, we'll process the full image and return as a single line
            # This is a limitation of the current TrOCR approach

            # Generate text
            generated_text = processor(
                pil_image,
                return_tensors="pt"
            ).pixel_values.to(model.device)

            generated_ids = model.generate(generated_text)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Create OCR line
            line = OCRLine(
                text=generated_text,
                confidence=1.0,  # TrOCR doesn't provide confidence scores
                bbox=BoundingBox(
                    x_min=0.0,
                    y_min=0.0,
                    x_max=float(image.width),
                    y_max=float(image.height),
                ),
                engine_name=self.name,
                language=context.config.language,
            )

            return [line]

        except Exception as exc:
            raise RecognitionError(f"TrOCR recognition failed: {exc}") from exc

    def _convert_to_pil(self, image: ImagePayload) -> Any:
        """Convert ImagePayload to PIL Image.

        Args:
            image: Input image payload.

        Returns:
            PIL Image object.

        Raises:
            RecognitionError: If conversion fails.
        """
        try:
            from PIL import Image as PILImage

            # Convert BGR to RGB if needed
            if image.color_space.value == "BGR":
                import cv2
                rgb_data = cv2.cvtColor(image.data, cv2.COLOR_BGR2RGB)
            else:
                rgb_data = image.data

            # Convert to PIL Image
            pil_image = PILImage.fromarray(rgb_data)
            return pil_image

        except ImportError:
            raise RecognitionError("PIL library is not installed")
        except Exception as exc:
            raise RecognitionError(f"Failed to convert image to PIL format: {exc}") from exc
