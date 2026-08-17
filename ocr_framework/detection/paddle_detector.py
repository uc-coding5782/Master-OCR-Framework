"""PaddleOCR-based text detector implementation."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

from ocr_framework.detection.base import TextDetector
from ocr_framework.exceptions import DetectionError
from ocr_framework.utils.gpu import resolve_use_gpu
from ocr_framework.utils.model_cache import ModelCache, paddle_cache_key
from ocr_framework.models.bbox import BoundingBox, Point, Polygon
from ocr_framework.models.image import ImagePayload
from ocr_framework.models.ocr_result import DetectionRegion
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.types import RegionType


class PaddleDetector(TextDetector):
    """PaddleOCR text detector using the DB detection model.

    This detector uses PaddleOCR's built-in detection capabilities to locate
    text regions in images. It supports orientation classification and can
    handle text at various angles.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the PaddleOCR detector.

        Args:
            lang: Language code for the detection model (e.g., 'en', 'ch', 'french').
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
        """Return the detector identifier."""
        return "paddle_detector"

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
                        enable_mkldnn=False,
                        show_log=False,
                    )
                except (TypeError, ValueError):
                    v3_kwargs: dict[str, Any] = {"lang": self._lang, "enable_mkldnn": False}
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

    def detect(
        self,
        image: ImagePayload,
        context: PipelineContext,
    ) -> list[DetectionRegion]:
        """Detect text regions in the image.

        Args:
            image: Input image payload.
            context: Pipeline execution context.

        Returns:
            List of detected text regions.

        Raises:
            DetectionError: If detection fails.
        """
        try:
            model = self._get_model()

            # Convert image data to the format expected by PaddleOCR
            img_array = self._prepare_image(image)

            # Run detection
            try:
                result = model.ocr(img_array, cls=True)
            except TypeError:
                result = model.ocr(img_array)

            # Extract detection regions
            regions = self._parse_detections(result)

            return regions

        except Exception as exc:
            raise DetectionError(f"PaddleOCR detection failed: {exc}") from exc

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

    def _parse_detections(self, result: list | None) -> list[DetectionRegion]:
        """Parse PaddleOCR output into DetectionRegion objects.

        Args:
            result: Raw output from PaddleOCR.ocr().

        Returns:
            List of DetectionRegion objects.
        """
        regions: list[DetectionRegion] = []

        if not result or result[0] is None:
            return regions

        if isinstance(result[0], dict):
            res_dict = result[0]
            polys = res_dict.get("dt_polys") or res_dict.get("rec_polys") or []
            for box_points in polys:
                points = tuple(Point(x=float(p[0]), y=float(p[1])) for p in box_points)
                polygon = Polygon(points=points)
                region = DetectionRegion(
                    bbox=polygon,
                    confidence=1.0,
                    region_type=RegionType.TEXT,
                )
                regions.append(region)
        else:
            for detection in result[0]:
                # PaddleOCR returns: (box_points, (text, confidence))
                box_points, _ = detection

                # Convert box points to Polygon
                points = tuple(Point(x=float(p[0]), y=float(p[1])) for p in box_points)
                polygon = Polygon(points=points)

                # Create detection region
                region = DetectionRegion(
                    bbox=polygon,
                    confidence=1.0,
                    region_type=RegionType.TEXT,
                )
                regions.append(region)

        return regions
