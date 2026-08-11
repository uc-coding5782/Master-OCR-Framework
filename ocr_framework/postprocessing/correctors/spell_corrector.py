"""Spell correction postprocessing corrector."""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False

from ocr_framework.models.page_result import PageResult

if TYPE_CHECKING:
    from ocr_framework.pipeline.context import PipelineContext


class SpellCorrector:
    """Apply spell correction to recognized text."""

    SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "pt", "ru"}

    def __init__(self, language: str = "en") -> None:
        """Initialize the spell corrector.

        Args:
            language: Language code for spell correction (e.g., 'en', 'es', 'fr').

        Raises:
            ImportError: If spellchecker is not installed.
        """
        if not SPELLCHECKER_AVAILABLE:
            raise ImportError(
                "spellchecker is not installed. Install it with: pip install pyspellchecker"
            )

        self._language = language
        self._spell_checker: SpellChecker | None = None

    @property
    def name(self) -> str:
        """Return the corrector identifier."""
        return "spell_corrector"

    def correct(self, page_result: PageResult, context: "PipelineContext") -> PageResult:
        """Apply spell correction to all lines in the page result.

        Args:
            page_result: Page result to correct.
            context: Pipeline execution context.

        Returns:
            Page result with corrected text.
        """
        _ = context  # Reserved for future use

        # Skip if language is not supported
        if self._language not in self.SUPPORTED_LANGUAGES:
            return page_result

        # Lazy-load spell checker
        if self._spell_checker is None:
            self._spell_checker = SpellChecker(language=self._language)

        # Correct each line
        for line in page_result.lines:
            line.text = self._correct_text(line.text)

        return page_result

    def _correct_text(self, text: str) -> str:
        """Correct spelling in a single line of text.

        Args:
            text: Input text to correct.

        Returns:
            Spell-corrected text.
        """
        words = text.split()
        corrected = []

        for word in words:
            prefix_len = 0
            while prefix_len < len(word) and not word[prefix_len].isalnum():
                prefix_len += 1
            suffix_len = 0
            while suffix_len < len(word) - prefix_len and not word[len(word) - 1 - suffix_len].isalnum():
                suffix_len += 1

            prefix = word[:prefix_len]
            suffix = word[len(word) - suffix_len :] if suffix_len > 0 else ""
            stripped = word[prefix_len : len(word) - suffix_len] if suffix_len > 0 else word[prefix_len:]

            # Skip empty or non-alphabetic words
            if not stripped or not stripped.isalpha():
                corrected.append(word)
                continue

            # Get correction
            fix = self._spell_checker.correction(stripped)
            corrected.append(prefix + (fix if fix else stripped) + suffix)

        return " ".join(corrected)
