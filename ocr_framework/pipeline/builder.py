"""Pipeline construction utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ocr_framework.config.profiles import get_profile
from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.detection.paddle_detector import PaddleDetector
from ocr_framework.loaders.image_loader import ImageLoader
from ocr_framework.pipeline.components import PipelineComponents
from ocr_framework.pipeline.runner import PipelineRunner
from ocr_framework.postprocessing.composite import CompositePostProcessor
from ocr_framework.postprocessing.correctors.spell_corrector import SpellCorrector
from ocr_framework.postprocessing.filters.confidence_filter import ConfidenceFilter
from ocr_framework.preprocessing.composite import CompositePreprocessor
from ocr_framework.preprocessing.steps.contrast import ContrastStep
from ocr_framework.preprocessing.steps.deskew import DeskewStep
from ocr_framework.preprocessing.steps.denoise import DenoiseStep
from ocr_framework.preprocessing.steps.upscale import UpscaleStep
from ocr_framework.recognition.paddle_recognizer import PaddleRecognizer

if TYPE_CHECKING:
    from ocr_framework.recognition.trocr_recognizer import TrOCRRecognizer


class PipelineBuilder:
    """Fluent builder for constructing a configured ``PipelineRunner``."""

    def __init__(self) -> None:
        self._config = FrameworkConfig()
        self._components = PipelineComponents()

    def with_config(self, config: FrameworkConfig) -> PipelineBuilder:
        """Apply a complete configuration object."""
        self._config = config
        return self

    def with_profile(self, profile: str) -> PipelineBuilder:
        """Set the configuration profile name."""
        self._config = get_profile(profile)
        return self

    def with_language(self, language: str) -> PipelineBuilder:
        """Set the OCR language code."""
        self._config.language = language
        return self

    def with_paddle_ocr(self) -> PipelineBuilder:
        """Configure the pipeline with PaddleOCR components.

        This sets up:
        - ImageLoader for loading image files
        - CompositePreprocessor with standard preprocessing steps
        - PaddleDetector for text detection
        - PaddleRecognizer for text recognition
        - CompositePostProcessor with confidence filtering and spell correction

        Returns:
            The builder instance for chaining.

        Raises:
            ImportError: If PaddleOCR or spellchecker are not installed.
        """
        # Configure loader
        self._components.loader = ImageLoader()

        # Configure preprocessing
        preprocessing_steps = [
            UpscaleStep(min_dim=1000),
            DenoiseStep(),
            DeskewStep(min_angle_threshold=0.5),
            ContrastStep(),
        ]
        self._components.preprocessor = CompositePreprocessor(steps=preprocessing_steps)

        # Configure detection
        self._components.detector = PaddleDetector(
            lang=self._config.language,
            use_angle_cls=True,
            use_gpu=self._config.use_gpu,
        )

        # Configure recognition
        self._components.recognizer = PaddleRecognizer(
            lang=self._config.language,
            use_angle_cls=True,
            use_gpu=self._config.use_gpu,
        )

        # Configure postprocessing
        postprocessors = [
            ConfidenceFilter(min_confidence=self._config.postprocessing.min_confidence),
        ]

        # Only add spell corrector if available
        try:
            postprocessors.append(SpellCorrector(language=self._config.language))
        except ImportError:
            # Spell checker not available, skip it
            pass

        self._components.postprocessor = CompositePostProcessor(processors=postprocessors)

        return self

    def with_trocr(
        self,
        model_name: str = "microsoft/trocr-base-handwritten",
    ) -> PipelineBuilder:
        """Configure the pipeline with TrOCR components.

        This sets up:
        - ImageLoader for loading image files
        - CompositePreprocessor with standard preprocessing steps
        - PaddleDetector for text detection (TrOCR doesn't have built-in detection)
        - TrOCRRecognizer for text recognition
        - CompositePostProcessor with confidence filtering

        Args:
            model_name: HuggingFace model name for TrOCR.

        Returns:
            The builder instance for chaining.

        Raises:
            ImportError: If transformers library is not installed.
        """
        from ocr_framework.recognition.trocr_recognizer import TrOCRRecognizer

        # Configure loader
        self._components.loader = ImageLoader()

        # Configure preprocessing
        preprocessing_steps = [
            UpscaleStep(min_dim=1000),
            DenoiseStep(),
            DeskewStep(min_angle_threshold=0.5),
            ContrastStep(),
        ]
        self._components.preprocessor = CompositePreprocessor(steps=preprocessing_steps)

        # Configure detection (use PaddleOCR for detection)
        self._components.detector = PaddleDetector(
            lang=self._config.language,
            use_angle_cls=True,
            use_gpu=self._config.use_gpu,
        )

        # Configure recognition with TrOCR
        self._components.recognizer = TrOCRRecognizer(
            model_name=model_name,
            use_gpu=self._config.use_gpu,
        )

        # Configure postprocessing (no spell correction for TrOCR)
        postprocessors = [
            ConfidenceFilter(min_confidence=self._config.postprocessing.min_confidence),
        ]
        self._components.postprocessor = CompositePostProcessor(processors=postprocessors)

        return self

    def with_components(self, components: PipelineComponents) -> PipelineBuilder:
        """Set custom pipeline components."""
        self._components = components
        return self

    def build(self) -> PipelineRunner:
        """Build and return a pipeline runner instance."""
        return PipelineRunner(config=self._config, components=self._components)
