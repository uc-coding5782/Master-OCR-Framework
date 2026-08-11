"""PaddleOCR-based text recognizer implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from ocr_framework.exceptions import RecognitionError
from ocr_framework.utils.gpu import resolve_use_gpu
from ocr_framework.utils.model_cache import ModelCache, paddle_cache_key
from ocr_framework.models.bbox import BoundingBox, Point, Polygon
from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion, OCRLine, OCRToken
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.recognition.base import TextRecognizer


class PaddleRecognizer(TextRecognizer):
    """PaddleOCR text recognizer using the CRNN recognition model.

    This recognizer uses PaddleOCR's built-in recognition capabilities to
    transcribe text from detected regions. It supports multiple languages
    and provides confidence scores for each recognized line.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the PaddleOCR recognizer.

        Args:
            lang: Language code for the recognition model (e.g., 'en', 'ch', 'french').
            use_angle_cls: Whether to use angle classification for rotated text.
            use_gpu: Whether to use GPU acceleration.

        Raises:
            ImportError: If PaddleOCR is not installed.
        """
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Install it with: "
                "pip install paddleocr paddlepaddle"
            )

        self._lang = lang
        self._use_angle_cls = use_angle_cls
        self._use_gpu = resolve_use_gpu(use_gpu)
        self._model: PaddleOCR | None = None

    @property
    def name(self) -> str:
        """Return the recognizer identifier."""
        return "paddle_recognizer"

    def _get_model(self) -> PaddleOCR:
        """Lazy-load the PaddleOCR model.

        Returns:
            The initialized PaddleOCR instance.
        """
        if self._model is None:
            key = paddle_cache_key(self._lang, self._use_angle_cls, self._use_gpu)

            def _create() -> PaddleOCR:
                try:
                    return PaddleOCR(
                        use_angle_cls=self._use_angle_cls,
                        lang=self._lang,
                        use_gpu=self._use_gpu,
                        show_log=False,
                    )
                except (TypeError, ValueError):
                    v3_kwargs: dict[str, Any] = {"lang": self._lang}
                    if self._use_angle_cls:
                        v3_kwargs["use_textline_orientation"] = True
                    if self._use_gpu:
                        v3_kwargs["device"] = "gpu"
                    try:
                        return PaddleOCR(**v3_kwargs)
                    except (TypeError, ValueError):
                        return PaddleOCR(lang=self._lang)

            self._model = ModelCache.get_or_create(key, _create)
        return self._model

    def recognize(
        self,
        image: ImagePayload,
        regions: list[DetectionRegion],
        context: PipelineContext,
    ) -> list[OCRLine]:
        """Recognize text from detected regions.

        Args:
            image: Input image payload.
            regions: Detected text regions.
            context: Pipeline execution context.

        Returns:
            List of recognized OCR lines.

        Raises:
            RecognitionError: If recognition fails.
        """
        try:
            model = self._get_model()

            # Convert image data to the format expected by PaddleOCR
            img_array = self._prepare_image(image)

            # Run full OCR (detection + recognition)
            try:
                result = model.ocr(img_array, cls=True)
            except TypeError:
                result = model.ocr(img_array)

            # Extract recognition results
            lines = self._parse_recognition_results(result, context.config.language)

            return lines

        except Exception as exc:
            raise RecognitionError(f"PaddleOCR recognition failed: {exc}") from exc

    def _prepare_image(self, image: ImagePayload) -> np.ndarray[Any, Any]:
        """Prepare image data for PaddleOCR.

        Args:
            image: Input image payload.

        Returns:
            NumPy array in the format expected by PaddleOCR.
        """
        # PaddleOCR expects BGR format
        if image.color_space.value != "BGR":
            # Convert color space if needed (implementation depends on current color space)
            # For now, assume the image is already in a compatible format
            pass

        return image.data

    def _parse_recognition_results(
        self,
        result: list | None,
        language: str,
    ) -> list[OCRLine]:
        """Parse PaddleOCR output into OCRLine objects.

        Args:
            result: Raw output from PaddleOCR.ocr().
            language: Language code for the recognition result.

        Returns:
            List of OCRLine objects.
        """
        lines: list[OCRLine] = []

        if not result or result[0] is None:
            return lines

        if isinstance(result[0], dict):
            res_dict = result[0]
            texts = res_dict.get("rec_texts") or []
            scores = res_dict.get("rec_scores") or []
            polys = res_dict.get("rec_polys") or res_dict.get("dt_polys") or []
            for i in range(len(texts)):
                box_points = polys[i] if i < len(polys) else []
                points = tuple(Point(x=float(p[0]), y=float(p[1])) for p in box_points)
                polygon = Polygon(points=points)
                line = OCRLine(
                    text=texts[i],
                    confidence=float(scores[i]) if i < len(scores) else 1.0,
                    bbox=polygon,
                    engine_name=self.name,
                    language=language,
                )
                lines.append(line)
        else:
            for ocr_result in result[0]:
                # PaddleOCR returns: (box_points, (text, confidence))
                box_points, (text, confidence) = ocr_result

                # Convert box points to Polygon
                points = tuple(Point(x=float(p[0]), y=float(p[1])) for p in box_points)
                polygon = Polygon(points=points)

                # Create OCR line
                line = OCRLine(
                    text=text,
                    confidence=float(confidence),
                    bbox=polygon,
                    engine_name=self.name,
                    language=language,
                )
                lines.append(line)

        return lines
