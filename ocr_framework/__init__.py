"""
OCR Framework — reusable, extensible OCR pipeline library.

Phase 1 provides domain models, configuration schemas, abstract interfaces,
and a minimal pipeline runner skeleton.

Phase 2 adds concrete implementations for PaddleOCR-based processing.

Phase 3 adds batch processing capabilities.

Phase 4 adds evaluation and benchmarking capabilities.

Phase 5 adds Microsoft TrOCR integration.

Phase 6 adds intelligent engine routing and fallback.

Phase 7 adds image quality analysis.

Phase 8 adds adaptive preprocessing based on quality analysis.

Phase 9 adds document intelligence (handwriting detection, language detection, document classification).

Phase 10 adds PDF and multi-page document support.

Phase 11 adds export framework (TXT, JSON, DOCX, Searchable PDF).

Phase 12 adds performance optimization (GPU detection, model cache, resource management).

Phase 13 adds FastAPI service for production OCR operations.

Version 1.0.0 — Production-ready OCR framework.
"""

from ocr_framework.config.schema import FrameworkConfig
from ocr_framework.exceptions import OCRFrameworkError
from ocr_framework.pipeline.builder import PipelineBuilder
from ocr_framework.pipeline.runner import PipelineRunner
from ocr_framework.quality.quality_analyzer import QualityAnalyzer

__version__ = "1.0.0"

__all__ = [
    # Core framework
    "FrameworkConfig",
    "OCRFrameworkError",
    "PipelineBuilder",
    "PipelineRunner",
    "QualityAnalyzer",
    "__version__",
]


def create_pipeline(
    profile: str = "default",
    config: FrameworkConfig | None = None,
) -> PipelineRunner:
    """Create a pipeline runner from a profile name or explicit config.

    Args:
        profile: Configuration profile name (e.g., 'default', 'document').
        config: Optional explicit configuration object.

    Returns:
        A configured PipelineRunner instance.
    """
    builder = PipelineBuilder()
    if config is not None:
        builder.with_config(config)
    else:
        builder.with_profile(profile)
    return builder.build()


def create_paddle_pipeline(
    language: str = "en",
    use_gpu: bool = False,
) -> PipelineRunner:
    """Create a pipeline runner with PaddleOCR components.

    Args:
        language: Language code for OCR (e.g., 'en', 'ch', 'french').
        use_gpu: Whether to use GPU acceleration.

    Returns:
        A PipelineRunner configured with PaddleOCR components.
    """
    builder = PipelineBuilder()
    builder.with_language(language)
    builder._config.use_gpu = use_gpu
    builder.with_paddle_ocr()
    return builder.build()


def create_trocr_pipeline(
    model_name: str = "microsoft/trocr-base-handwritten",
    use_gpu: bool = False,
) -> PipelineRunner:
    """Create a pipeline runner with TrOCR components.

    Args:
        model_name: HuggingFace model name for TrOCR.
        use_gpu: Whether to use GPU acceleration.

    Returns:
        A PipelineRunner configured with TrOCR components.
    """
    builder = PipelineBuilder()
    builder._config.use_gpu = use_gpu
    builder.with_trocr(model_name=model_name)
    return builder.build()


def create_batch_processor(
    pipeline: PipelineRunner,
    output_dir: str,
    silent: bool = False,
):
    """Create a batch processor for processing multiple files.

    Args:
        pipeline: Configured pipeline runner for individual file processing.
        output_dir: Target output directory for results.
        silent: If True, suppress console output.

    Returns:
        A BatchProcessor instance.
    """
    from pathlib import Path
    from ocr_framework.batch.batch_processor import BatchProcessor

    return BatchProcessor(
        pipeline=pipeline,
        config=pipeline.config,
        output_dir=Path(output_dir),
        silent=silent,
    )
