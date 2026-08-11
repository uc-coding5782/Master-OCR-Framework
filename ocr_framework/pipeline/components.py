"""Pipeline dependency container."""

from __future__ import annotations

from dataclasses import dataclass

from ocr_framework.engines.detector import TextDetector
from ocr_framework.engines.recognizer import TextRecognizer
from ocr_framework.exporters.base import Exporter
from ocr_framework.loaders.base import DocumentLoader
from ocr_framework.postprocessing.base import PostProcessor
from ocr_framework.preprocessing.base import Preprocessor


@dataclass
class PipelineComponents:
    """Injectable dependencies required by ``PipelineRunner`` stage methods.

    Each field is optional at construction time. Individual stage methods
    raise ``NotImplementedError`` when their corresponding dependency has not
    been provided.

    Attributes:
        loader: Loads input files into ``Document`` instances.
        preprocessor: Transforms page images before detection.
        detector: Locates text regions within a page image.
        recognizer: Converts detected regions into ``OCRLine`` objects.
        postprocessor: Cleans and filters page-level OCR output.
        exporter: Serializes ``DocumentResult`` objects to external formats.
    """

    loader: DocumentLoader | None = None
    preprocessor: Preprocessor | None = None
    detector: TextDetector | None = None
    recognizer: TextRecognizer | None = None
    postprocessor: PostProcessor | None = None
    exporter: Exporter | None = None
