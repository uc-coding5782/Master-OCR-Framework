"""Pipeline orchestration for the OCR framework."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.exceptions import PipelineError
from ocr_framework.models.page_result import DocumentResult, PageResult
from ocr_framework.observability.logging import get_logger
from ocr_framework.performance.resource_manager import ResourceManager
from ocr_framework.pipeline.components import PipelineComponents
from ocr_framework.pipeline.context import PipelineContext
from ocr_framework.routing.confidence_manager import ConfidenceManager
from ocr_framework.routing.engine_selector import EngineSelector
from ocr_framework.routing.retry_manager import RetryManager

logger = get_logger("pipeline.runner")


class PipelineRunner:
    """Orchestrate OCR pipeline stages from input path to ``DocumentResult``.

    The runner depends on abstract component interfaces rather than concrete
    OCR implementations. Each stage method delegates to its configured
    dependency and raises ``NotImplementedError`` when that dependency is
    missing.

    Attributes:
        config: Active framework configuration.
        components: Injectable pipeline dependencies.
    """

    def __init__(
        self,
        config: FrameworkConfig | None = None,
        components: PipelineComponents | None = None,
    ) -> None:
        """Initialize the pipeline runner.

        Args:
            config: Optional framework configuration. Defaults are used when
                omitted.
            components: Optional dependency container. Missing dependencies
                cause stage methods to raise ``NotImplementedError``.
        """
        self.config = config or FrameworkConfig()
        self.components = components or PipelineComponents()

        # Initialize routing components
        self._engine_selector = EngineSelector(self.config)
        self._confidence_manager = ConfidenceManager(self.config)
        self._retry_manager = RetryManager(self.config)

        # Resource manager for cleanup and memory optimization
        self._resource_manager = ResourceManager()

    def run(
        self,
        input_path: Path | str,
        export_destination: Path | str | None = None,
    ) -> DocumentResult:
        """Execute the full OCR pipeline for a single input document.

        Args:
            input_path: Path to the input image or document file.
            export_destination: Optional export target path. When provided,
                ``export()`` is invoked after OCR completes.

        Returns:
            The final ``DocumentResult`` produced by the pipeline.

        Raises:
            PipelineError: If a stage fails or the pipeline ends without a
                result.
            NotImplementedError: If a required component has not been
                configured.
        """
        context = PipelineContext(
            input_path=Path(input_path),
            config=self.config,
            export_destination=(
                Path(export_destination) if export_destination is not None else None
            ),
        )

        logger.info("Starting pipeline", extra={"path": str(input_path)})

        try:
            context = self.load(context)
            context = self._process_pages(context)
            context = self._assemble_document_result(context)
            if context.export_destination is not None:
                context = self.export(context)
        except NotImplementedError:
            raise
        except PipelineError:
            raise
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise PipelineError(f"Pipeline execution failed: {exc}") from exc

        if context.document_result is None:
            raise PipelineError("Pipeline completed without producing a DocumentResult.")

        logger.info(
            "Pipeline completed",
            extra={
                "path": str(input_path),
                "pages": context.document_result.page_count,
            },
        )
        return context.document_result

    def load(self, context: PipelineContext) -> PipelineContext:
        """Load a document from disk into the pipeline context.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with ``document`` populated.

        Raises:
            NotImplementedError: If no ``DocumentLoader`` was configured.
        """
        # Try PDF loader first for PDF/TIFF files
        if self.components.loader is None:
            raise NotImplementedError("DocumentLoader is not configured.")

        # Check if file is PDF/TIFF and we have a PDF loader available
        if context.input_path.suffix.lower() in {".pdf", ".tif", ".tiff"}:
            try:
                from ocr_framework.loaders.pdf_loader import PDFLoader

                pdf_loader = PDFLoader(config=self.config)
                if pdf_loader.supports(context.input_path):
                    context.document = pdf_loader.load(context.input_path)
                    return context
            except ImportError:
                # Fall back to regular loader if PDF loader not available
                pass

        # Use regular loader
        context.document = self.components.loader.load(context.input_path)
        return context

    def preprocess(self, context: PipelineContext) -> PipelineContext:
        """Apply preprocessing to the current page image.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with ``processed_image`` populated.

        Raises:
            NotImplementedError: If no ``Preprocessor`` was configured.
            PipelineError: If no active page image is available.
        """
        preprocessor = self.components.preprocessor
        if preprocessor is None:
            raise NotImplementedError("Preprocessor is not configured.")
        if context.current_page is None:
            raise PipelineError("Current page must be set before preprocessing.")

        source_image = context.current_page.image
        context.processed_image = preprocessor.process(source_image, context)
        context.current_page.preprocessing_trace.append(
            {"preprocessor": preprocessor.name}
        )
        return context

    def detect(self, context: PipelineContext) -> PipelineContext:
        """Detect text regions on the active page image.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with ``detections`` populated.

        Raises:
            NotImplementedError: If no ``TextDetector`` was configured.
            PipelineError: If no active image is available.
        """
        detector = self.components.detector
        if detector is None:
            raise NotImplementedError("TextDetector is not configured.")

        image = context.active_image
        if image is None:
            raise PipelineError("An active page image is required for detection.")

        context.detections = detector.detect(image, context)
        return context

    def recognize(self, context: PipelineContext) -> PipelineContext:
        """Recognize text from detected regions on the active page.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with ``recognized_lines`` populated.

        Raises:
            NotImplementedError: If no ``TextRecognizer`` was configured.
            PipelineError: If no active image is available.
        """
        recognizer = self.components.recognizer
        if recognizer is None:
            raise NotImplementedError("TextRecognizer is not configured.")

        image = context.active_image
        if image is None:
            raise PipelineError("An active page image is required for recognition.")

        context.recognized_lines = recognizer.recognize(
            image,
            context.detections,
            context,
        )
        return context

    def postprocess(self, context: PipelineContext) -> PipelineContext:
        """Apply postprocessing to the current page result.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with the processed page appended to
            ``page_results``.

        Raises:
            NotImplementedError: If no ``PostProcessor`` was configured.
            PipelineError: If no page result is available to postprocess.
        """
        postprocessor = self.components.postprocessor
        if postprocessor is None:
            raise NotImplementedError("PostProcessor is not configured.")

        page_result = context.current_page_result or self._build_page_result(context)
        context.current_page_result = postprocessor.process(page_result, context)
        context.page_results.append(context.current_page_result)
        return context

    def export(self, context: PipelineContext) -> PipelineContext:
        """Export the assembled document result.

        Args:
            context: Current pipeline execution context.

        Returns:
            Updated context with ``export_report`` populated.

        Raises:
            NotImplementedError: If no ``Exporter`` was configured.
            PipelineError: If no document result or destination is available.
        """
        exporter = self.components.exporter
        if exporter is None:
            raise NotImplementedError("Exporter is not configured.")
        if context.document_result is None:
            raise PipelineError("Document result must exist before export.")
        if context.export_destination is None:
            raise PipelineError("Export destination must be set before export.")

        context.export_report = exporter.export(
            context.document_result,
            context.export_destination,
        )
        return context

    def run_with_routing(
        self,
        input_path: Path | str,
        export_destination: Path | str | None = None,
    ) -> DocumentResult:
        """Execute the OCR pipeline with intelligent routing and fallback.

        This method automatically retries with alternative engines if the
        primary engine produces low-confidence results.

        Args:
            input_path: Path to the input image or document file.
            export_destination: Optional export target path.

        Returns:
            The final ``DocumentResult`` produced by the pipeline.

        Raises:
            PipelineError: If all retry attempts fail.
        """
        context = PipelineContext(
            input_path=Path(input_path),
            config=self.config,
            export_destination=(
                Path(export_destination) if export_destination is not None else None
            ),
        )

        try:
            context = self.load(context)
            context = self._process_pages_with_routing(context)
            context = self._assemble_document_result(context)
            if context.export_destination is not None:
                context = self.export(context)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(f"Pipeline execution failed: {exc}") from exc

        if context.document_result is None:
            raise PipelineError("Pipeline completed without producing a DocumentResult.")

        return context.document_result

    def cleanup(self) -> None:
        """Release resources held by the pipeline.

        Triggers cleanup callbacks registered with the ``ResourceManager``
        and forces garbage collection to reclaim memory. Safe to call
        multiple times.
        """
        self._resource_manager.cleanup()

    def _process_pages_with_routing(self, context: PipelineContext) -> PipelineContext:
        """Run per-page stages with intelligent routing and fallback.

        Args:
            context: Context with a loaded ``document``.

        Returns:
            Updated context with page-level outputs populated.

        Raises:
            PipelineError: If the document has not been loaded.
        """
        if context.document is None:
            raise PipelineError("Document must be loaded before page processing.")

        for page_index, page in enumerate(context.document.pages):
            context.current_page_index = page_index
            context.current_page = page
            context.processed_image = None
            context.detections = []
            context.recognized_lines = []
            context.current_page_result = None
            context.routing_decisions = []

            # Select initial engine
            selected_engine = self._engine_selector.select_engine(
                page_result=None,
                image_metadata=page.image.metadata,
            )

            # Process with routing and retry logic
            attempt = 0
            max_retries = self._retry_manager.get_max_retries()
            last_error = None

            while attempt <= max_retries:
                try:
                    # Swap recognizer if needed
                    if attempt > 0:
                        self._swap_recognizer(selected_engine)

                    # Run the pipeline stages
                    context = self.preprocess(context)
                    context = self.detect(context)
                    context = self.recognize(context)
                    context.current_page_result = self._build_page_result(context)

                    # Check if result is acceptable
                    if self._confidence_manager.is_acceptable(context.current_page_result):
                        break

                    # Check if we should retry
                    if not self._retry_manager.should_retry(context.current_page_result, attempt):
                        break

                    # Select fallback engine
                    selected_engine = self._retry_manager.get_next_engine(selected_engine, attempt)
                    attempt += 1

                except Exception as exc:
                    last_error = exc
                    attempt += 1
                    if attempt > max_retries:
                        raise

            # Postprocess the final result
            context = self.postprocess(context)

        return context

    def _swap_recognizer(self, engine_name: str) -> None:
        """Swap the recognizer for a different engine.

        Args:
            engine_name: Engine identifier to switch to.
        """
        if engine_name == "paddle":
            from ocr_framework.recognition.paddle_recognizer import PaddleRecognizer

            self.components.recognizer = PaddleRecognizer(
                lang=self.config.language,
                use_angle_cls=True,
                use_gpu=self.config.use_gpu,
            )
        elif engine_name == "trocr":
            from ocr_framework.recognition.trocr_recognizer import TrOCRRecognizer

            self.components.recognizer = TrOCRRecognizer(
                model_name="microsoft/trocr-base-handwritten",
                use_gpu=self.config.use_gpu,
            )

    def _process_pages(self, context: PipelineContext) -> PipelineContext:
        """Run per-page stages for every loaded document page.

        Args:
            context: Context with a loaded ``document``.

        Returns:
            Updated context with page-level outputs populated.

        Raises:
            PipelineError: If the document has not been loaded.
        """
        if context.document is None:
            raise PipelineError("Document must be loaded before page processing.")

        for page_index, page in enumerate(context.document.pages):
            context.current_page_index = page_index
            context.current_page = page
            context.processed_image = None
            context.detections = []
            context.recognized_lines = []
            context.current_page_result = None

            timings: dict[str, float] = {}

            start = time.perf_counter()
            context = self.preprocess(context)
            timings["preprocess"] = time.perf_counter() - start

            start = time.perf_counter()
            context = self.detect(context)
            timings["detect"] = time.perf_counter() - start

            start = time.perf_counter()
            context = self.recognize(context)
            timings["recognize"] = time.perf_counter() - start

            context.current_page_result = self._build_page_result(context)
            context.current_page_result.timings = timings

            start = time.perf_counter()
            context = self.postprocess(context)
            timings["postprocess"] = time.perf_counter() - start
            if context.page_results:
                context.page_results[-1].timings = timings

            logger.debug(
                "Page processed",
                extra={
                    "page_index": page_index,
                    "duration_ms": sum(timings.values()) * 1000,
                },
            )

        return context

    def _assemble_document_result(self, context: PipelineContext) -> PipelineContext:
        """Build the final ``DocumentResult`` from accumulated page outputs.

        Args:
            context: Context with populated ``page_results``.

        Returns:
            Updated context with ``document_result`` populated.

        Raises:
            PipelineError: If the source document is missing.
        """
        if context.document is None:
            raise PipelineError("Document must be loaded before assembling results.")

        context.document_result = DocumentResult(
            document=context.document,
            pages=list(context.page_results),
            metadata=dict(context.metadata),
        )
        return context

    def _build_page_result(self, context: PipelineContext) -> PageResult:
        """Create a page result from the current context state.

        Args:
            context: Context containing page-level OCR outputs.

        Returns:
            A ``PageResult`` populated from current detections and lines.

        Raises:
            PipelineError: If no active page is set.
        """
        if context.current_page is None:
            raise PipelineError("Current page must be set before building results.")

        # Get recognizer name from components
        recognizer_name = ""
        if self.components.recognizer is not None:
            recognizer_name = self.components.recognizer.name

        aggregate_confidence = _calculate_aggregate_confidence(context.recognized_lines)

        return PageResult(
            page_index=context.current_page_index,
            lines=list(context.recognized_lines),
            detections=list(context.detections),
            engine_used=recognizer_name,
            routing_decisions=list(context.routing_decisions),
            aggregate_confidence=aggregate_confidence,
            metadata={"language": context.config.language},
        )


def _calculate_aggregate_confidence(lines: list) -> float:
    """Compute a simple mean line confidence.

    Args:
        lines: Recognized OCR lines.

    Returns:
        Mean confidence, or ``0.0`` when no lines are present.
    """
    if not lines:
        return 0.0
    return sum(line.confidence for line in lines) / len(lines)