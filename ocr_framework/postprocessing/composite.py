"""Composite postprocessor that chains multiple filters and correctors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ocr_framework.exceptions import PipelineError
from ocr_framework.models.page_result import PageResult
from ocr_framework.postprocessing.base import PostProcessor

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class CompositePostProcessor(PostProcessor):
    """Postprocessor that applies multiple filters and correctors in sequence."""

    def __init__(self, processors: list[Any]) -> None:
        """Initialize the composite postprocessor.

        Args:
            processors: List of filter/corrector objects with a process() method.
        """
        self._processors = processors

    @property
    def name(self) -> str:
        """Return the postprocessor identifier."""
        return "composite"

    def process(
        self,
        page_result: PageResult,
        context: "PipelineContext",
    ) -> PageResult:
        """Apply all postprocessing steps in sequence.

        Args:
            page_result: Input page result.
            context: Pipeline execution context.

        Returns:
            Processed page result after all steps.

        Raises:
            PipelineError: If any postprocessing step fails.
        """
        processed_result = page_result

        for processor in self._processors:
            try:
                # Check if it's a filter or corrector
                if hasattr(processor, "filter"):
                    processed_result = processor.filter(processed_result, context)
                elif hasattr(processor, "correct"):
                    processed_result = processor.correct(processed_result, context)
                elif hasattr(processor, "process"):
                    processed_result = processor.process(processed_result, context)
                else:
                    raise PipelineError(
                        f"Postprocessor '{processor.name}' has no valid processing method"
                    )
            except PipelineError:
                raise
            except Exception as exc:
                raise PipelineError(
                    f"Postprocessing step '{processor.name}' failed: {exc}"
                ) from exc

        return processed_result
