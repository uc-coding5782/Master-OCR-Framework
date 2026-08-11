"""
Postprocessing: cleans up raw OCR output to improve final accuracy.
"""

from spellchecker import SpellChecker


def filter_low_confidence(results: list[dict], min_confidence: float = 0.5) -> list[dict]:
    """Drop OCR lines below a confidence threshold (likely garbage)."""
    return [r for r in results if r["confidence"] >= min_confidence]


def correct_spelling(text: str, lang: str = "en") -> str:
    """
    Run basic spell correction on English text.
    Note: pyspellchecker only supports a handful of languages (en, es, fr, de, pt, ru).
    For other languages, skip this step or swap in a language-specific corrector.
    """
    supported = {"en", "es", "fr", "de", "pt", "ru"}
    if lang not in supported:
        return text  # no-op for unsupported languages

    spell = SpellChecker(language=lang)
    words = text.split()
    corrected = []
    for w in words:
        prefix_len = 0
        while prefix_len < len(w) and not w[prefix_len].isalnum():
            prefix_len += 1
        suffix_len = 0
        while suffix_len < len(w) - prefix_len and not w[len(w) - 1 - suffix_len].isalnum():
            suffix_len += 1

        prefix = w[:prefix_len]
        suffix = w[len(w) - suffix_len :] if suffix_len > 0 else ""
        stripped = w[prefix_len : len(w) - suffix_len] if suffix_len > 0 else w[prefix_len:]

        if not stripped or not stripped.isalpha():
            corrected.append(w)
            continue
        fix = spell.correction(stripped)
        corrected.append(prefix + (fix if fix else stripped) + suffix)
    return " ".join(corrected)


def clean_text(results: list[dict], min_confidence: float = 0.5, lang: str = "en",
                spell_correct: bool = True) -> str:
    """Full postprocessing pipeline: filter -> join -> (optional) spell correct."""
    filtered = filter_low_confidence(results, min_confidence)
    joined = "\n".join(r["text"] for r in filtered)
    if spell_correct:
        joined = "\n".join(correct_spelling(line, lang) for line in joined.split("\n"))
    return joined
