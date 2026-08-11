"""Configuration schemas for the OCR framework."""

from __future__ import annotations

from dataclasses import dataclass, field

from ocr_framework.types import ExportFormat, Metadata, PreprocessingMode


@dataclass
class LoaderConfig:
    """Settings for document loaders."""

    pdf_dpi: int = 300


@dataclass
class PreprocessingConfig:
    """Settings for the preprocessing stage."""

    mode: PreprocessingMode = PreprocessingMode.INTELLIGENT
    steps: list[str] = field(
        default_factory=lambda: ["upscale", "denoise", "deskew", "contrast"]
    )
    intelligent_thresholds: Metadata = field(
        default_factory=lambda: {"blur": 0.4, "skew_deg": 0.5}
    )


@dataclass
class RoutingConfig:
    """Settings for engine routing and confidence thresholds."""

    primary_engine: str = "paddle"
    fallback_chain: list[str] = field(default_factory=lambda: ["trocr"])
    min_confidence: float = 0.55
    rules: list[str] = field(default_factory=lambda: ["confidence", "language"])


@dataclass
class PostprocessingConfig:
    """Settings for postprocessing filters and correctors."""

    steps: list[str] = field(
        default_factory=lambda: ["confidence_filter", "normalizer", "spell_corrector"]
    )
    spell_check_langs: list[str] = field(
        default_factory=lambda: ["en", "fr", "de", "es", "pt", "ru"]
    )
    min_confidence: float = 0.5


@dataclass
class ExportConfig:
    """Settings for result exporters."""

    formats: list[ExportFormat] = field(
        default_factory=lambda: [ExportFormat.JSON, ExportFormat.TXT]
    )
    include_boxes: bool = True
    output_dir: str | None = None


@dataclass
class BatchConfig:
    """Settings for batch processing."""

    workers: int = 4
    continue_on_error: bool = True
    output_format: str = "txt"


@dataclass
class FrameworkConfig:
    """Top-level configuration object for pipeline execution."""

    profile: str = "default"
    language: str = "en"
    use_gpu: bool = False
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    metadata: Metadata = field(default_factory=dict)
