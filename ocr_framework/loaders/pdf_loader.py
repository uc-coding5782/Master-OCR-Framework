"""PDF document loader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ocr_framework.exceptions import LoaderError
from ocr_framework.loaders.base import DocumentLoader
from ocr_framework.models.document import Document, Page
from ocr_framework.models.image import ImagePayload
from ocr_framework.pdf.pdf_loader import PDFLoader as PDFDocumentLoader
from ocr_framework.types import ColorSpace, Metadata

if TYPE_CHECKING:
    import cv2


class PDFLoader(DocumentLoader):
    """Load PDF and TIFF documents into framework Document objects.

    Supports multi-page PDF and TIFF files, extracting each page
    as a separate Page in the Document.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".tif", ".tiff"}

    def __init__(self, config: Any | None = None) -> None:
        """Initialize the PDF loader.

        Args:
            config: Framework configuration with PDF settings.
        """
        self._pdf_loader = PDFDocumentLoader(config)

    def supports(self, path: Path) -> bool:
        """Check if the loader supports the given file path.

        Args:
            path: Candidate input file path.

        Returns:
            True if the file has a supported PDF/TIFF extension.
        """
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, path: Path, options: Metadata | None = None) -> Document:
        """Load a PDF or TIFF file into a Document.

        Args:
            path: Input PDF/TIFF file path.
            options: Optional loader-specific settings.

        Returns:
            A Document with one Page per PDF/TIFF page.

        Raises:
            LoaderError: If the file cannot be loaded.
        """
        _ = options  # Reserved for future use

        if not path.exists():
            raise LoaderError(f"PDF file not found: {path}")

        try:
            # Load PDF document
            pdf_doc = self._pdf_loader.load(path)

            # Convert PDF pages to framework Pages
            pages = []
            for pdf_page in pdf_doc.pages:
                # Convert image array to BGR color space (OpenCV format)
                try:
                    import cv2

                    if len(pdf_page.image.shape) == 3:
                        # Convert RGB to BGR
                        image_data = cv2.cvtColor(pdf_page.image, cv2.COLOR_RGB2BGR)
                    else:
                        image_data = pdf_page.image
                except ImportError:
                    # If cv2 not available, use as-is
                    image_data = pdf_page.image

                # Create ImagePayload
                image = ImagePayload(
                    data=image_data,
                    width=pdf_page.width,
                    height=pdf_page.height,
                    channels=image_data.shape[2] if len(image_data.shape) == 3 else 1,
                    color_space=ColorSpace.BGR,
                    dpi=pdf_page.dpi,
                    metadata=pdf_page.metadata,
                )

                # Create Page
                page = Page(page_index=pdf_page.page_number, image=image)

                pages.append(page)

            # Create Document
            document = Document(
                pages=pages,
                source_path=path,
                mime_type=self._get_mime_type(path),
            )

            return document

        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to load PDF {path}: {exc}") from exc

    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for the PDF file.

        Args:
            path: PDF file path.

        Returns:
            MIME type string.
        """
        ext = path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }
        return mime_map.get(ext, "application/octet-stream")
