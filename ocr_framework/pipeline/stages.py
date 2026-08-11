"""Pipeline stage definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ocr_framework.pipeline.context import PipelineContext


class PipelineStage(ABC):
    """A single step in the OCR processing pipeline."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique stage identifier."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Run the stage and return the updated context."""


class LoadStage(PipelineStage):
    """Load an input file into a ``Document``."""

    @property
    def name(self) -> str:
        return "load"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("LoadStage is not implemented yet.")


class PreprocessStage(PipelineStage):
    """Apply preprocessing to the current page image."""

    @property
    def name(self) -> str:
        return "preprocess"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("PreprocessStage is not implemented yet.")


class DetectStage(PipelineStage):
    """Detect text regions on the current page."""

    @property
    def name(self) -> str:
        return "detect"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("DetectStage is not implemented yet.")


class RecognizeStage(PipelineStage):
    """Recognize text from detected regions."""

    @property
    def name(self) -> str:
        return "recognize"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("RecognizeStage is not implemented yet.")


class PostprocessStage(PipelineStage):
    """Apply postprocessing to recognized page results."""

    @property
    def name(self) -> str:
        return "postprocess"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("PostprocessStage is not implemented yet.")


class ExportStage(PipelineStage):
    """Export the final document result."""

    @property
    def name(self) -> str:
        return "export"

    def execute(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError("ExportStage is not implemented yet.")
