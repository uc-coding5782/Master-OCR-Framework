"""Language detection for document intelligence."""

from __future__ import annotations


class LanguageDetector:
    """Detect document language using character frequency analysis.

    This detector uses character frequency heuristics to estimate
    the primary language of the text. It can be extended with
    proper language detection libraries for higher accuracy.
    """

    def detect(self, text: str) -> str:
        """Detect the primary language of the text.

        Args:
            text: Input text string.

        Returns:
            Language code (e.g., 'en', 'unknown').
        """
        if not text:
            return "unknown"

        # Check for non-ASCII characters
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        total_count = len(text)

        if total_count == 0:
            return "unknown"

        # If mostly ASCII, assume English
        if non_ascii_count / total_count < 0.1:
            return "en"

        # Otherwise, return unknown (would need proper language detection library)
        return "unknown"

    def detect_batch(self, texts: list[str]) -> dict:
        """Detect languages for multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary with language counts.
        """
        languages = [self.detect(text) for text in texts]
        from collections import Counter
        counts = Counter(languages)

        return dict(counts)

    def get_confidence(self, text: str, language: str) -> float:
        """Get confidence score for language detection.

        Args:
            text: Input text string.
            language: Detected language.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not text or language == "unknown":
            return 0.0

        if language == "en":
            non_ascii_count = sum(1 for c in text if ord(c) > 127)
            total_count = len(text)

            if total_count == 0:
                return 0.0

            confidence = 1.0 - (non_ascii_count / total_count)
            return float(confidence)

        return 0.5  # Medium confidence for other languages
