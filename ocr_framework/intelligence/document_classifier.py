"""Document type classifier for document intelligence."""

from __future__ import annotations

import re


class DocumentClassifier:
    """Classify document type using heuristic analysis.

    This classifier uses text patterns and keywords to estimate
    the document type (receipt, invoice, form, book, etc.).
    """

    def __init__(self) -> None:
        """Initialize the document classifier."""
        # Keywords for different document types
        self._keywords = {
            "receipt": [
                "total", "amount", "cash", "card", "payment", "change",
                "subtotal", "tax", "item", "quantity", "price", "store",
                "shop", "market", "grocery", "pharmacy",
            ],
            "invoice": [
                "invoice", "bill to", "ship to", "due date", "terms",
                "account", "invoice number", "po number", "vendor",
                "supplier", "client", "company", "llc", "inc",
            ],
            "form": [
                "name", "address", "phone", "email", "date of birth",
                "signature", "field", "check box", "application",
                "form", "questionnaire", "survey",
            ],
            "book": [
                "chapter", "page", "isbn", "publisher", "author",
                "copyright", "edition", "volume", "table of contents",
                "index", "bibliography",
            ],
        }

    def classify(self, text: str) -> str:
        """Classify the document type based on text content.

        Args:
            text: Input text string.

        Returns:
            Document type (e.g., 'receipt', 'invoice', 'form', 'book', 'document').
        """
        if not text:
            return "document"

        text_lower = text.lower()

        # Score each document type based on keyword matches
        scores = {}
        for doc_type, keywords in self._keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score

        # Find document type with highest score
        if not scores:
            return "document"

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # If no strong matches, return generic document
        if max_score == 0:
            return "document"

        return max_type

    def get_confidence(self, text: str, doc_type: str) -> float:
        """Get confidence score for document classification.

        Args:
            text: Input text string.
            doc_type: Detected document type.

        Returns:
            Confidence score (0.0-1.0).
        """
        if not text or doc_type == "document":
            return 0.0

        text_lower = text.lower()
        keywords = self._keywords.get(doc_type, [])

        if not keywords:
            return 0.0

        matches = sum(1 for keyword in keywords if keyword in text_lower)
        confidence = min(matches / len(keywords), 1.0)

        return float(confidence)

    def classify_batch(self, texts: list[str]) -> dict:
        """Classify multiple documents.

        Args:
            texts: List of text strings.

        Returns:
            Dictionary with document type counts.
        """
        types = [self.classify(text) for text in texts]
        from collections import Counter
        counts = Counter(types)

        return dict(counts)
